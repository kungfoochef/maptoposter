"""
create_map_stl.py — Map to 3D Print Layer Generator

Companion to create_map_poster.py. Fetches the same OpenStreetMap geodata
(reusing the shared cache) and produces per-layer SVG files + an OpenSCAD
.scad file for multi-color FDM 3D printing.

Usage:
    python create_map_stl.py --city "Williamsville" --country "USA" --theme ocean
    python create_map_stl.py --city "Paris" --country "France" --latitude 48.8566 --longitude 2.3522
    python create_map_stl.py --list-themes
"""

import argparse
import asyncio
import json
import os
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — must come before pyplot import
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

import osmnx as ox
from font_management import load_fonts
from geopandas import GeoDataFrame
from geopy.geocoders import Nominatim
from lat_lon_parser import parse as parse_latlon
from networkx import MultiDiGraph
from shapely.affinity import affine_transform
from shapely.geometry import LineString, Point, Polygon as ShapelyPolygon, box
from shapely.ops import unary_union

# =============================================================================
# CONSTANTS
# =============================================================================

CACHE_DIR_PATH = os.environ.get("CACHE_DIR", "cache")
CACHE_DIR = Path(CACHE_DIR_PATH)
CACHE_DIR.mkdir(exist_ok=True)

THEMES_DIR = "themes"
STL_OUTPUT_DIR = "stl_output"
FILE_ENCODING = "utf-8"

# Road classification groups (matching create_map_poster.py exactly)
ROADS_MAJOR_TYPES = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
}
ROADS_MINOR_TYPES = {
    "tertiary", "tertiary_link", "residential",
    "living_street", "unclassified",
}

# Buffer radii in projected meters (half the effective road width)
BUFFER_MAJOR_M = 25.0
BUFFER_MINOR_M = 12.0

# SVG path data large-file warning threshold (bytes)
SVG_WARN_BYTES = 5_000_000


# =============================================================================
# HELPERS
# =============================================================================

def _is_latin_script(text: str) -> bool:
    """Return True if text is primarily Latin script (for letter-spacing in title)."""
    if not text:
        return True
    latin_count = sum(1 for c in text if c.isalpha() and ord(c) < 0x250)
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return True
    return (latin_count / total_alpha) > 0.8


# =============================================================================
# CACHE HELPERS  (verbatim from create_map_poster.py)
# =============================================================================

class CacheError(Exception):
    """Raised when a cache operation fails."""


def _cache_path(key: str) -> str:
    safe = key.replace(os.sep, "_")
    return os.path.join(CACHE_DIR, f"{safe}.pkl")


def cache_get(key: str):
    try:
        path = _cache_path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CacheError(f"Cache read failed: {e}") from e


def cache_set(key: str, value):
    try:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        path = _cache_path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise CacheError(f"Cache write failed: {e}") from e


# =============================================================================
# THEME HELPERS  (verbatim from create_map_poster.py)
# =============================================================================

def load_theme(theme_name: str = "terracotta") -> dict:
    theme_file = os.path.join(THEMES_DIR, f"{theme_name}.json")
    if not os.path.exists(theme_file):
        print(f"  Theme '{theme_file}' not found. Using default terracotta theme.")
        return {
            "name": "Terracotta",
            "description": "Mediterranean warmth - burnt orange and clay tones on cream",
            "bg": "#F5EDE4",
            "text": "#8B4513",
            "gradient_color": "#F5EDE4",
            "water": "#A8C4C4",
            "parks": "#E8E0D0",
            "road_motorway": "#A0522D",
            "road_primary": "#B8653A",
            "road_secondary": "#C9846A",
            "road_tertiary": "#D9A08A",
            "road_residential": "#E5C4B0",
            "road_default": "#D9A08A",
        }
    with open(theme_file, "r", encoding=FILE_ENCODING) as f:
        theme = json.load(f)
        print(f"  Loaded theme: {theme.get('name', theme_name)}")
        if "description" in theme:
            print(f"  {theme['description']}")
        return theme


def get_available_themes() -> list:
    themes_path = Path(THEMES_DIR)
    if not themes_path.exists():
        return []
    return sorted(p.stem for p in themes_path.glob("*.json"))


# =============================================================================
# GEOCODING  (verbatim from create_map_poster.py)
# =============================================================================

def get_coordinates(city: str, country: str) -> tuple:
    coords_key = f"coords_{city.lower()}_{country.lower()}"
    cached = cache_get(coords_key)
    if cached:
        print(f"  Using cached coordinates for {city}, {country}")
        return cached

    print("  Looking up coordinates...")
    geolocator = Nominatim(user_agent="city_map_poster", timeout=10)
    time.sleep(1)

    try:
        location = geolocator.geocode(f"{city}, {country}")
    except Exception as e:
        raise ValueError(f"Geocoding failed for {city}, {country}: {e}") from e

    if asyncio.iscoroutine(location):
        try:
            location = asyncio.run(location)
        except RuntimeError as exc:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError(
                    "Geocoder returned a coroutine while an event loop is already running."
                ) from exc
            location = loop.run_until_complete(location)

    if location:
        addr = getattr(location, "address", None)
        print(f"  Found: {addr}" if addr else "  Found location")
        print(f"  Coordinates: {location.latitude}, {location.longitude}")
        try:
            cache_set(coords_key, (location.latitude, location.longitude))
        except CacheError as e:
            print(e)
        return (location.latitude, location.longitude)

    raise ValueError(f"Could not find coordinates for {city}, {country}")


# =============================================================================
# DATA FETCHING  (same cache keys as create_map_poster.py → shared cache)
# =============================================================================

def fetch_graph(point: tuple, dist: float) -> MultiDiGraph | None:
    lat, lon = point
    graph_key = f"graph_{lat}_{lon}_{dist}"
    cached = cache_get(graph_key)
    if cached is not None:
        print("  Using cached street network")
        return cast(MultiDiGraph, cached)

    try:
        g = ox.graph_from_point(
            point, dist=dist, dist_type="bbox",
            network_type="all", truncate_by_edge=True,
        )
        time.sleep(0.5)
        try:
            cache_set(graph_key, g)
        except CacheError as e:
            print(e)
        return g
    except Exception as e:
        print(f"  OSMnx error fetching graph: {e}")
        return None


def fetch_features(point: tuple, dist: float, tags: dict, name: str) -> GeoDataFrame | None:
    lat, lon = point
    tag_str = "_".join(tags.keys())
    features_key = f"{name}_{lat}_{lon}_{dist}_{tag_str}"
    cached = cache_get(features_key)
    if cached is not None:
        print(f"  Using cached {name}")
        return cast(GeoDataFrame, cached)

    try:
        data = ox.features_from_point(point, tags=tags, dist=dist)
        time.sleep(0.3)
        try:
            cache_set(features_key, data)
        except CacheError as e:
            print(e)
        return data
    except Exception as e:
        print(f"  OSMnx error fetching {name}: {e}")
        return None


# =============================================================================
# GEOMETRY PROCESSING
# =============================================================================

def project_center(lat: float, lon: float, target_crs: str) -> tuple:
    """Project WGS84 center point into metric CRS. Returns (x, y) in meters."""
    projected, _ = ox.projection.project_geometry(
        Point(lon, lat), crs="EPSG:4326", to_crs=target_crs
    )
    return (projected.x, projected.y)


def compute_bbox(center_proj: tuple, dist: float) -> tuple:
    """Return square (minx, miny, maxx, maxy) centered on center_proj ± dist meters."""
    cx, cy = center_proj
    return (cx - dist, cy - dist, cx + dist, cy + dist)


def extract_polygon_geometry(gdf: GeoDataFrame, bbox_geom, target_crs: str):
    """Project GDF, filter to polygons, clip to bbox, union all. Returns geometry or None."""
    if gdf is None or gdf.empty:
        return None

    # Project to metric CRS
    try:
        gdf_proj = ox.projection.project_gdf(gdf)
    except Exception:
        try:
            gdf_proj = gdf.to_crs(target_crs)
        except Exception as e:
            print(f"    Warning: could not project GDF: {e}")
            return None

    # Filter to polygon types only
    poly_mask = gdf_proj.geometry.type.isin(["Polygon", "MultiPolygon"])
    polys = gdf_proj[poly_mask]
    if polys.empty:
        return None

    # Clip and union
    clipped = []
    for geom in polys.geometry:
        try:
            intersection = bbox_geom.intersection(geom)
            if not intersection.is_empty:
                clipped.append(intersection)
        except Exception:
            continue

    if not clipped:
        return None

    result = unary_union(clipped)
    return result if not result.is_empty else None


def extract_road_geometry(g_proj: MultiDiGraph, road_types: set, buffer_m: float, bbox_geom):
    """Buffer road edges of given types, clip to bbox, union. Returns geometry or None."""
    lines = []
    for u, v, data in g_proj.edges(data=True):
        highway = data.get("highway", "unclassified")
        if isinstance(highway, list):
            highway = highway[0] if highway else "unclassified"

        if highway in road_types:
            if "geometry" in data:
                lines.append(data["geometry"])
            else:
                u_node = g_proj.nodes[u]
                v_node = g_proj.nodes[v]
                lines.append(LineString([(u_node["x"], u_node["y"]),
                                         (v_node["x"], v_node["y"])]))

    if not lines:
        return None

    buffered = [line.buffer(buffer_m, cap_style=1, join_style=1) for line in lines]
    unioned = unary_union(buffered)
    clipped = bbox_geom.intersection(unioned)
    return clipped if not clipped.is_empty else None


# =============================================================================
# OPENSCAD COORDINATE HELPER
# =============================================================================

def compute_openscad_position(
    lat: float, lon: float, bbox: tuple, print_size_mm: float, target_crs: str
) -> tuple:
    """
    Convert a lat/lon to OpenSCAD XY coordinates in mm.

    OpenSCAD's coordinate system is cartesian (Y increases upward/northward),
    matching the projected CRS — so this is a plain scale+translate with no Y flip.
    This differs from the SVG transform which does flip Y.
    """
    minx, miny, maxx, _ = bbox
    scale = print_size_mm / (maxx - minx)
    projected, _ = ox.projection.project_geometry(
        Point(lon, lat), crs="EPSG:4326", to_crs=target_crs
    )
    return ((projected.x - minx) * scale, (projected.y - miny) * scale)


# =============================================================================
# GEOMETRY CLEANING
# =============================================================================

# Minimum polygon area to keep (m²). Fragments smaller than this are invisible
# at city scale and cause degenerate CGAL faces in OpenSCAD's F6 render.
_MIN_POLY_AREA_M2 = 0.5

def _clean_geom(geom, simplify_tol_m: float = 0.5):
    """
    Clean shapely geometry before writing to SVG to avoid OpenSCAD CGAL errors:

      "ERROR: The given mesh is not closed! Unable to convert to CGAL_Nef_Polyhedron."

    Four-step process (all in projected meter space):
      1. buffer(0) — repairs self-intersecting or degenerate rings that shapely's
         unary_union can produce at road junction seams.
      2. buffer(+eps).buffer(-eps) — merges regions that touch at a single point or
         share a zero-width throat. When extruded, touching-at-a-point geometry
         produces non-manifold edges that CGAL Nef rejects. eps=0.5m = 0.008mm at
         200mm/6000m scale — completely invisible. Two separate polygons must be
         < 1.0m apart to be accidentally merged, which won't happen in road geometry.
      3. simplify() — removes near-duplicate vertices.
      4. Filter tiny fragments — drop any Polygon part with area < _MIN_POLY_AREA_M2.
    """
    if geom is None or geom.is_empty:
        return geom
    try:
        geom = geom.buffer(0)
        geom = geom.buffer(0.5).buffer(-0.5)   # merge touching-at-a-point regions
        geom = geom.simplify(simplify_tol_m, preserve_topology=True)
    except Exception:
        pass

    # Filter sub-polygons below minimum area
    def _filter_parts(g):
        from shapely.geometry import MultiPolygon as _MPoly
        gtype = g.geom_type if not g.is_empty else "empty"
        if gtype == "Polygon":
            return g if g.area >= _MIN_POLY_AREA_M2 else None
        if gtype in ("MultiPolygon", "GeometryCollection"):
            kept = [_filter_parts(part) for part in g.geoms]
            kept = [p for p in kept if p is not None and not p.is_empty]
            if not kept:
                return None
            return unary_union(kept)
        return g  # LineString, Point, etc. — pass through unchanged

    try:
        cleaned = _filter_parts(geom)
        if cleaned is not None and not cleaned.is_empty:
            return cleaned
    except Exception:
        pass
    return geom  # return original if cleaning fails


# =============================================================================
# SVG WRITING
# =============================================================================

def build_coordinate_transform(bbox: tuple, print_size_mm: float) -> tuple:
    """
    Build affine transform params [a, b, d, e, xoff, yoff] for
    shapely.affinity.affine_transform() that maps projected meters to SVG mm.

    SVG has Y increasing downward, so we flip Y.
    1 SVG unit = 1 mm (OpenSCAD reads SVG mm attributes directly).
    """
    minx, miny, maxx, maxy = bbox
    span = maxx - minx  # = dist * 2
    scale = print_size_mm / span
    # new_x = scale * x  - minx * scale
    # new_y = -scale * y + maxy * scale  (Y flip)
    return (scale, 0, 0, -scale, -minx * scale, maxy * scale)


def _ring_to_path_cmds(coords) -> str:
    pts = list(coords)
    # Remove the closing duplicate that shapely appends
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if not pts:
        return ""
    cmd = f"M {pts[0][0]:.4f} {pts[0][1]:.4f}"
    for x, y in pts[1:]:
        cmd += f" L {x:.4f} {y:.4f}"
    cmd += " Z"
    return cmd


def _collect_polygons(geom):
    """Yield all Polygon parts from any geometry type."""
    if geom is None or geom.is_empty:
        return
    gtype = geom.geom_type
    if gtype == "Polygon":
        yield geom
    elif gtype in ("MultiPolygon", "GeometryCollection"):
        for part in geom.geoms:
            yield from _collect_polygons(part)


def geometry_to_svg_path_data(geom) -> str:
    """Convert any Shapely geometry to SVG path 'd' string using evenodd fill."""
    if geom is None or geom.is_empty:
        return ""

    parts = []
    for poly in _collect_polygons(geom):
        # Exterior ring
        parts.append(_ring_to_path_cmds(poly.exterior.coords))
        # Interior rings (holes)
        for interior in poly.interiors:
            parts.append(_ring_to_path_cmds(interior.coords))

    return " ".join(p for p in parts if p)


def write_layer_svg(
    output_path: Path,
    geom,
    transform_params: tuple,
    print_size_mm: float,
    fill_color: str,
    layer_name: str,
    is_background: bool = False,
) -> bool:
    """
    Write a single layer SVG file.

    Returns True if geometry was non-empty, False if empty (still writes a valid SVG).
    """
    size = f"{print_size_mm:.4f}"
    header = (
        f'<?xml version="1.0" encoding="utf-8"?>\n'
        f'<!-- Map 3D Print Layer: {layer_name} -->\n'
        f'<svg xmlns="http://www.w3.org/2000/svg"\n'
        f'     viewBox="0 0 {size} {size}"\n'
        f'     width="{size}mm"\n'
        f'     height="{size}mm">\n'
    )

    if is_background:
        body = f'  <rect x="0" y="0" width="{size}" height="{size}" fill="{fill_color}"/>\n'
        content = header + body + "</svg>\n"
        output_path.write_text(content, encoding=FILE_ENCODING)
        return True

    if geom is None or geom.is_empty:
        content = (
            header
            + f'  <!-- WARNING: No geometry found for layer "{layer_name}". Empty layer. -->\n'
            + "</svg>\n"
        )
        output_path.write_text(content, encoding=FILE_ENCODING)
        print(f"    Warning: no geometry for layer '{layer_name}' — SVG will be empty")
        return False

    # Apply coordinate transform
    transformed = affine_transform(geom, transform_params)

    # Write one <path> per Polygon rather than one mega-path.
    # A single giant path with thousands of sub-rings and evenodd fill causes
    # OpenSCAD CGAL to fail ("mesh not closed") because adjacent polygon rings
    # that share an edge produce zero-width regions when evenodd double-counts
    # those shared edges. Per-polygon paths avoid all cross-polygon interaction.
    path_lines = []
    for poly in _collect_polygons(transformed):
        d = _ring_to_path_cmds(poly.exterior.coords)
        for interior in poly.interiors:
            d += " " + _ring_to_path_cmds(interior.coords)
        if d:
            path_lines.append(
                f'  <path d="{d}"\n'
                f'        fill="{fill_color}"\n'
                f'        fill-rule="evenodd"\n'
                f'        stroke="none"/>'
            )

    if not path_lines:
        content = (
            header
            + f'  <!-- WARNING: Empty path data for layer "{layer_name}". -->\n'
            + "</svg>\n"
        )
        output_path.write_text(content, encoding=FILE_ENCODING)
        return False

    body = "\n".join(path_lines) + "\n"
    content = header + body + "\n</svg>\n"
    output_path.write_text(content, encoding=FILE_ENCODING)

    # Warn on very large SVGs
    size_bytes = len(content.encode(FILE_ENCODING))
    if size_bytes > SVG_WARN_BYTES:
        print(
            f"    Warning: '{layer_name}' SVG is {size_bytes // 1_000_000}MB. "
            f"OpenSCAD may be slow. Consider a smaller --distance."
        )

    return True


def _textpath_to_svg_d(tp, viewbox_height: float) -> str:
    """
    Convert a matplotlib TextPath to an SVG path d string with Y-axis flipped.

    matplotlib uses Y-up; SVG uses Y-down.  Flip: svg_y = viewbox_height - mpl_y.
    Handles MOVETO, LINETO, CURVE3 (quadratic), CURVE4 (cubic), and CLOSEPOLY codes.
    """
    from matplotlib.path import Path as MplPath

    verts = tp.vertices
    codes = tp.codes
    n = len(codes)
    if n == 0:
        return ""

    parts = []
    i = 0
    while i < n:
        code = int(codes[i])
        x  = float(verts[i][0])
        y  = float(verts[i][1])
        sy = viewbox_height - y  # Y flip

        if code == MplPath.MOVETO:  # 1
            parts.append(f"M {x:.4f} {sy:.4f}")
            i += 1

        elif code == MplPath.LINETO:  # 2
            parts.append(f"L {x:.4f} {sy:.4f}")
            i += 1

        elif code == MplPath.CURVE3:  # 3 — quadratic bezier: control + end (2 verts)
            cx, csy = x, sy
            i += 1
            if i < n:
                ex  = float(verts[i][0])
                esy = viewbox_height - float(verts[i][1])
                parts.append(f"Q {cx:.4f} {csy:.4f} {ex:.4f} {esy:.4f}")
                i += 1

        elif code == MplPath.CURVE4:  # 4 — cubic bezier: c1 + c2 + end (3 verts)
            c1x, c1sy = x, sy
            i += 1
            if i + 1 < n:
                c2x  = float(verts[i][0])
                c2sy = viewbox_height - float(verts[i][1])
                i += 1
                ex  = float(verts[i][0])
                esy = viewbox_height - float(verts[i][1])
                parts.append(f"C {c1x:.4f} {c1sy:.4f} {c2x:.4f} {c2sy:.4f} {ex:.4f} {esy:.4f}")
                i += 1
            else:
                i += 1

        elif code == MplPath.CLOSEPOLY:  # 79
            parts.append("Z")
            i += 1

        else:
            i += 1

    return " ".join(parts)


def generate_title_svg(
    output_path: Path,
    display_city: str,
    display_country: str,
    theme: dict,
    print_size_mm: float,
    font_files: dict | None = None,
) -> bool:
    """
    Render the map title (city name, divider bar, country) as SVG bezier paths.

    Uses matplotlib.textpath.TextPath to convert text to path data in a coordinate
    space we fully control — no matplotlib figure or <g transform> elements are
    emitted, so OpenSCAD's import() can extrude the paths without transform support.

    Coordinate system
    -----------------
    ViewBox: 864 × 864 pt  (= 12 in × 72 pt/in, identical to poster's figsize=12
    at scale_factor=1.0).  TextPath without a figure defaults to 72 DPI, so its
    vertex coordinates are in pt and map 1-to-1 into the viewBox.
    OpenSCAD scales the viewBox to width/height = print_size_mm × print_size_mm.

    Returns True on success.
    """
    text_color = theme.get("text", "#1A1A1A")

    # ViewBox size in pt: 12 in × 72 pt/in = 864 pt
    # TextPath at 72 DPI uses the same pt scale, so no extra scaling is needed.
    VB = 864.0

    # Font sizes in pt — identical to create_map_poster.py (scale_factor = 1.0)
    base_main_pt = 60.0
    base_sub_pt  = 22.0
    city_char_count = len(display_city)
    if city_char_count > 10:
        font_size_main = max(base_main_pt * (10.0 / city_char_count), 10.0)
    else:
        font_size_main = base_main_pt

    if font_files:
        fp_main = FontProperties(fname=font_files["bold"],  size=font_size_main)
        fp_sub  = FontProperties(fname=font_files["light"], size=base_sub_pt)
    else:
        fp_main = FontProperties(family="sans-serif", weight="bold",  size=font_size_main)
        fp_sub  = FontProperties(family="sans-serif", weight="light", size=base_sub_pt)

    # Letter-space Latin text — matching poster behavior exactly
    spaced_city = "  ".join(list(display_city.upper())) if _is_latin_script(display_city) else display_city

    # Y positions as fractions from bottom in Y-up (matplotlib) space.
    # Y_DIV_BOT is shifted 1.5mm lower than the poster default (0.1225 → 0.115).
    Y_CITY    = 0.14
    Y_DIV_BOT = 0.115
    Y_COUNTRY = 0.10

    svg_parts = []

    def _add_text(text: str, fp: FontProperties, y_center_frac: float) -> None:
        """Append SVG path d data for text, centered horizontally at y_center_frac."""
        if not text:
            return
        try:
            # Measure bounding box with text placed at origin
            tp0 = TextPath((0, 0), text, prop=fp)
            bb  = tp0.get_extents()
            w   = bb.x1 - bb.x0
            h   = bb.y1 - bb.y0
            if w <= 0 or h <= 0:
                return
            # Shift origin so the bbox center lands at (VB/2, y_center_frac * VB)
            x_origin = VB / 2.0 - w / 2.0 - bb.x0
            y_origin = y_center_frac * VB - h / 2.0 - bb.y0
            tp = TextPath((x_origin, y_origin), text, prop=fp)
            d = _textpath_to_svg_d(tp, VB)
            if d:
                svg_parts.append(d)
        except Exception as exc:
            print(f"    Warning: could not render text '{text}': {exc}")

    _add_text(spaced_city,             fp_main, Y_CITY)
    _add_text(display_country.upper(), fp_sub,  Y_COUNTRY)

    # Divider bar — filled rectangle as a closed polygon path.
    # In Y-up space: spans y=Y_DIV_BOT*VB (bottom) to (Y_DIV_BOT+0.005)*VB (top).
    # In SVG Y-down: top ↔ bottom swap.
    div_bot_mpl = Y_DIV_BOT * VB
    div_top_mpl = (Y_DIV_BOT + 0.005) * VB
    div_top_svg = VB - div_top_mpl   # smaller SVG y = higher on page
    div_bot_svg = VB - div_bot_mpl
    lx  = 0.4 * VB
    lx2 = 0.6 * VB
    svg_parts.append(
        f"M {lx:.4f} {div_top_svg:.4f} "
        f"L {lx2:.4f} {div_top_svg:.4f} "
        f"L {lx2:.4f} {div_bot_svg:.4f} "
        f"L {lx:.4f} {div_bot_svg:.4f} Z"
    )

    all_d = " ".join(svg_parts)
    svg_content = (
        f'<?xml version="1.0" encoding="utf-8"?>\n'
        f'<!-- Map 3D Print Layer: title -->\n'
        f'<svg xmlns="http://www.w3.org/2000/svg"\n'
        f'     viewBox="0 0 {VB:.4f} {VB:.4f}"\n'
        f'     width="{print_size_mm}mm"\n'
        f'     height="{print_size_mm}mm">\n'
        f'  <path d="{all_d}"\n'
        f'        fill="{text_color}"\n'
        f'        fill-rule="evenodd"\n'
        f'        stroke="none"/>\n'
        f'</svg>\n'
    )

    output_path.write_text(svg_content, encoding=FILE_ENCODING)
    return True


def _generate_title_geometry(
    display_city: str,
    display_country: str,
    font_files: dict | None,
    bbox: tuple,
    print_size_mm: float,
):
    """
    Convert the title text (city, divider bar, country) into shapely geometry
    in projected map coordinates.

    TextPath bezier curves are approximated via to_polygons() and each sub-polygon
    is mapped from the 864pt viewbox Y-up space into the projected CRS so the
    result can be unioned with water/parks/roads for physical layer support.

    Returns a single (Multi)Polygon or None if nothing could be generated.
    """
    VB = 864.0
    minx, miny, maxx, maxy = bbox
    span = maxx - minx  # square bbox → span == maxy - miny

    def _vb_to_proj(pt_x: float, pt_y: float) -> tuple:
        """Map a point from 864pt Y-up viewbox space to projected meter space."""
        return (minx + pt_x * span / VB, miny + pt_y * span / VB)

    # Font setup — identical to generate_title_svg()
    base_main_pt = 60.0
    base_sub_pt  = 22.0
    city_char_count = len(display_city)
    font_size_main = max(base_main_pt * (10.0 / city_char_count), 10.0) if city_char_count > 10 else base_main_pt

    if font_files:
        fp_main = FontProperties(fname=font_files["bold"],  size=font_size_main)
        fp_sub  = FontProperties(fname=font_files["light"], size=base_sub_pt)
    else:
        fp_main = FontProperties(family="sans-serif", weight="bold",  size=font_size_main)
        fp_sub  = FontProperties(family="sans-serif", weight="light", size=base_sub_pt)

    spaced_city = "  ".join(list(display_city.upper())) if _is_latin_script(display_city) else display_city

    # Separate outer rings from holes.  TrueType spec: outer contours are CW
    # (negative signed area via shoelace); enclosed counters/holes are CCW (positive).
    # This is opposite the PDF/PostScript convention.  Naively unioning all sub-polygons
    # fills holes (e.g. counters of 'O','A','B','P','R'), so we must subtract them.
    shapes_outer: list = []
    shapes_holes: list = []

    def _signed_area(pts) -> float:
        """Shoelace formula. Negative → CW (TrueType outer ring). Positive → CCW (hole)."""
        n = len(pts)
        return sum(
            pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
            for i in range(n)
        ) / 2.0

    def _add_text_geom(text: str, fp: FontProperties, y_center_frac: float) -> None:
        if not text:
            return
        try:
            tp0 = TextPath((0, 0), text, prop=fp)
            bb  = tp0.get_extents()
            w, h = bb.x1 - bb.x0, bb.y1 - bb.y0
            if w <= 0 or h <= 0:
                return
            x_origin = VB / 2.0 - w / 2.0 - bb.x0
            y_origin = y_center_frac * VB - h / 2.0 - bb.y0
            tp = TextPath((x_origin, y_origin), text, prop=fp)
            # to_polygons() approximates bezier curves as polylines and splits on
            # each MOVETO — one array per letter contour or hole component.
            for poly_pts in tp.to_polygons():
                if len(poly_pts) < 3:
                    continue
                proj_pts = [_vb_to_proj(p[0], p[1]) for p in poly_pts]
                sa = _signed_area(proj_pts)
                try:
                    poly = ShapelyPolygon(proj_pts).buffer(0)
                    if not poly.is_empty:
                        # TrueType outer contours are CW = negative signed area.
                        # CCW (positive area) = hole (counter, bowl of 'O','A','B','P','R').
                        (shapes_outer if sa <= 0 else shapes_holes).append(poly)
                except Exception:
                    pass
        except Exception:
            pass

    _add_text_geom(spaced_city,             fp_main, 0.14)
    _add_text_geom(display_country.upper(), fp_sub,  0.10)

    # Divider bar — thin filled rectangle (always an outer ring).
    # Y_DIV_BOT shifted 1.5mm lower than poster default (0.1225 → 0.115).
    div_bot_vb = 0.115 * VB
    div_top_vb = (0.115 + 0.005) * VB
    lx, lx2 = 0.4 * VB, 0.6 * VB
    try:
        shapes_outer.append(ShapelyPolygon([
            _vb_to_proj(lx,  div_bot_vb),
            _vb_to_proj(lx2, div_bot_vb),
            _vb_to_proj(lx2, div_top_vb),
            _vb_to_proj(lx,  div_top_vb),
        ]))
    except Exception:
        pass

    valid_outer = [s for s in shapes_outer if s is not None and not s.is_empty]
    if not valid_outer:
        return None
    result = unary_union(valid_outer)
    valid_holes = [s for s in shapes_holes if s is not None and not s.is_empty]
    if valid_holes:
        result = result.difference(unary_union(valid_holes))
    return result if not result.is_empty else None


# =============================================================================
# OPENSCAD GENERATION
# =============================================================================

def build_layer_configs(theme: dict, base_height_mm: float, layer_step_mm: float) -> list:
    """Build the ordered list of layer configs with Z positions pre-calculated."""
    return [
        {
            "index": 1,
            "name": "background",
            "filename": "01_background.svg",
            "color": theme.get("bg", "#CCCCCC"),
            "color_label": "background / base plate",
            "z_start": 0.0,
            "height": base_height_mm,
            "is_background": True,
        },
        {
            "index": 2,
            "name": "water",
            "filename": "02_water.svg",
            "color": theme.get("water", "#88BBCC"),
            "color_label": "water features",
            "z_start": base_height_mm,
            "height": layer_step_mm,
            "is_background": False,
        },
        {
            "index": 3,
            "name": "parks",
            "filename": "03_parks.svg",
            "color": theme.get("parks", "#99CC99"),
            "color_label": "parks / green spaces",
            "z_start": base_height_mm + layer_step_mm,
            "height": layer_step_mm,
            "is_background": False,
        },
        {
            "index": 4,
            "name": "roads_minor",
            "filename": "04_roads_minor.svg",
            "color": theme.get("road_tertiary", "#AAAAAA"),
            "color_label": "minor roads (tertiary, residential)",
            "z_start": base_height_mm + 2 * layer_step_mm,
            "height": layer_step_mm,
            "is_background": False,
        },
        {
            "index": 5,
            "name": "roads_major",
            "filename": "05_roads_major.svg",
            "color": theme.get("road_primary", "#888888"),
            "color_label": "major roads (motorway, primary, secondary)",
            "z_start": base_height_mm + 3 * layer_step_mm,
            "height": layer_step_mm,
            "is_background": False,
        },
    ]


def generate_scad_file(
    output_path: Path,
    layer_configs: list,
    print_size_mm: float,
    base_height_mm: float,
    layer_step_mm: float,
    city: str,
    country: str,
    theme_name: str,
    lat: float,
    lon: float,
    dist: float,
    timestamp: str,
    display_city: str,
    display_country: str,
    marker_pos: tuple | None = None,
    marker_text: str = "You are here",
    marker_text_position: str = "right",
    title_color: str = "#1A1A1A",
    marker_color: str = "#CC0000",
    scad_font: str = "Arial",
) -> None:
    z_title  = base_height_mm + 4 * layer_step_mm
    z_marker = base_height_mm + 5 * layer_step_mm
    total_height = z_marker + layer_step_mm if marker_pos else z_title + layer_step_mm

    # Build filament change guide (SVG layers + title + optional marker)
    all_layers = []
    for cfg in layer_configs:
        all_layers.append((cfg["z_start"], cfg["color"], cfg["color_label"], cfg["index"] == 1))
    all_layers.append((z_title, title_color, "title text", False))
    if marker_pos:
        all_layers.append((z_marker, marker_color, "you-are-here marker", False))

    change_lines = []
    for z, color, label, is_first in all_layers:
        z_str = f"{z:.2f}mm"
        prefix = "Start print  " if is_first else "Color change "
        change_lines.append(f" *   {z_str:<7} {prefix} → load {color}  ({label})")
    filament_guide = "\n".join(change_lines)

    # Each layer overlaps the one below by a small epsilon so there are no coincident
    # faces at the Z boundaries.  Plane-to-plane touching solids produce internal seams
    # in the exported STL that slicer watertight-mesh validators (Cura, Meshmixer) flag
    # as non-manifold, even though CGAL's own check passes.  0.01 mm is invisible at
    # print scale but sufficient to convert a face-touching join into a solid union.
    _OV = 0.01  # mm overlap applied at each layer interface

    # SVG layer modules
    layer_modules = []
    for cfg in layer_configs:
        z = max(0.0, cfg['z_start'] - _OV)          # sink into layer below (clamp ≥ 0)
        h = cfg['height'] + (_OV if cfg['z_start'] == 0.0 else 2 * _OV)  # extend both ends
        layer_modules.append(
            f"module layer_{cfg['name']}() {{\n"
            f"    color(\"{cfg['color']}\") {{\n"
            f"        translate([0, 0, {z:.4f}])\n"
            f"            map_layer(\"layers/{cfg['filename']}\", {h:.4f});\n"
            f"    }}\n"
            f"}}"
        )

    # Star polygon module (5-pointed, defined once, reused by marker)
    star_module = """\
module star_2d(outer_r, inner_r, num_points = 5) {
    step      = 360 / num_points;
    half_step = step / 2;
    polygon([
        for (i = [0 : num_points - 1])
        each [
            [ outer_r * cos(90 + i * step),            outer_r * sin(90 + i * step) ],
            [ inner_r * cos(90 + i * step + half_step), inner_r * sin(90 + i * step + half_step) ]
        ]
    ]);
}"""

    # Title module — imports the matplotlib-rendered SVG (text as bezier paths).
    # Typography is identical to the poster: spaced city name, divider, country.
    title_module = (
        f"module layer_title() {{\n"
        f"    color(\"{title_color}\") {{\n"
        f"        translate([0, 0, {z_title - _OV:.4f}])\n"
        f"            map_layer(\"layers/06_title.svg\", {layer_step_mm + _OV:.4f});\n"
        f"    }}\n"
        f"}}"
    )

    # Marker module — star + label at specified OpenSCAD position (only if marker_pos given)
    marker_module = ""
    if marker_pos:
        mx, my = marker_pos
        outer_r_mm = print_size_mm * 0.025
        gap = outer_r_mm * 1.4  # space between star edge and label

        pos = marker_text_position.lower()
        if pos == "right":
            tx, ty, halign, valign = mx + gap, my, "left", "center"
        elif pos == "left":
            tx, ty, halign, valign = mx - gap, my, "right", "center"
        elif pos == "above":
            tx, ty, halign, valign = mx, my + gap, "center", "bottom"
        else:  # "below"
            tx, ty, halign, valign = mx, my - gap, "center", "top"

        marker_module = (
            f"\nmodule layer_marker() {{\n"
            f"    color(\"{marker_color}\") {{\n"
            f"        translate([0, 0, {z_marker - _OV:.4f}]) {{\n"
            f"            // Star\n"
            f"            translate([{mx:.4f}, {my:.4f}, 0])\n"
            f"                linear_extrude(height = layer_step + {_OV})\n"
            f"                    star_2d(outer_r = print_size * 0.025,\n"
            f"                            inner_r = print_size * 0.01);\n"
            f"            // Marker label\n"
            f"            translate([{tx:.4f}, {ty:.4f}, 0])\n"
            f"                linear_extrude(height = layer_step + {_OV})\n"
            f"                    text(\"{marker_text}\", size = print_size * 0.028,\n"
            f"                         font = \"{scad_font}\",\n"
            f"                         halign = \"{halign}\", valign = \"{valign}\");\n"
            f"        }}\n"
            f"    }}\n"
            f"}}"
        )

    # Union assembly
    svg_calls   = "\n".join(f"    layer_{cfg['name']}();" for cfg in layer_configs)
    extra_calls = "\n    layer_title();"
    if marker_pos:
        extra_calls += "\n    layer_marker();"

    # Echo statements
    echo_lines = []
    for z, color, label, is_first in all_layers:
        action = "Start" if is_first else "Change"
        echo_lines.append(f'echo(str("  {z:.2f}mm  {action}: load {color}  ({label})"));')
    echo_block = "\n".join(echo_lines)

    scad = f"""\
/*
 * 3D Print Map: {city}, {country}
 * Theme: {theme_name}
 * Center: {lat:.6f}, {lon:.6f}
 * Radius: {dist}m  |  Print size: {print_size_mm}mm x {print_size_mm}mm
 * Generated by create_map_stl.py on {timestamp}
 *
 * FILAMENT CHANGE GUIDE
 * =====================
{filament_guide}
 *
 * SLICER SETUP
 * ============
 * Add filament change (M600) commands at each Z height listed above.
 *   PrusaSlicer : right-click layer marker on the layer slider → "Add color change"
 *   Cura        : Extensions → Post Processing → Add Script → FilamentChange
 */

// ============================================================
// PARAMETERS
// ============================================================

print_size   = {print_size_mm};     // mm — square print dimensions
base_height  = {base_height_mm};    // mm — background plate height
layer_step   = {layer_step_mm};     // mm — height of each feature layer
total_height = {total_height:.4f};  // mm — full stack height

// ============================================================
// SVG FILE PATHS (relative — keep layers/ folder next to this file)
// ============================================================

{chr(10).join(f'f_{cfg["name"]:15s} = "layers/{cfg["filename"]}";' for cfg in layer_configs)}
f_title           = "layers/06_title.svg";

// ============================================================
// HELPER MODULE
// ============================================================

/*
 * Import and extrude one SVG layer.
 * SVG viewBox is {print_size_mm}mm x {print_size_mm}mm — 1 SVG unit = 1mm.
 * OpenSCAD reads the SVG width/height attributes in mm directly.
 * No scale() transform needed.
 */
module map_layer(svg_file, height) {{
    linear_extrude(height = height) {{
        import(svg_file, center = false);
    }}
}}

// 5-pointed star polygon (used by layer_marker)
{star_module}

// ============================================================
// LAYER MODULES
// ============================================================

{chr(10).join(chr(10) + m for m in layer_modules)}

{title_module}
{marker_module}

// ============================================================
// MAIN ASSEMBLY
// ============================================================

union() {{
{svg_calls}{extra_calls}
}}

// ============================================================
// PRINT INFO (shown in OpenSCAD console on render)
// ============================================================

echo("============================================================");
echo("Map 3D Print: {city}, {country}");
echo("============================================================");
echo(str("Print size  : {print_size_mm} x {print_size_mm} mm"));
echo(str("Total height: ", total_height, " mm"));
echo("");
echo("Filament changes:");
{echo_block}
echo("============================================================");
"""

    output_path.write_text(scad, encoding=FILE_ENCODING)


# =============================================================================
# ORCHESTRATION
# =============================================================================

def create_map_stl(
    city: str,
    country: str,
    point: tuple,
    dist: float,
    theme_name: str,
    theme: dict,
    print_size_mm: float,
    base_height_mm: float,
    layer_step_mm: float,
    display_city: str = "",
    display_country: str = "",
    marker_lat: float | None = None,
    marker_lon: float | None = None,
    marker_text: str = "You are here",
    marker_text_position: str = "right",
    title_color: str = "#1A1A1A",
    marker_color: str = "#CC0000",
) -> Path:
    lat, lon = point

    # Match the poster script's compensated_dist formula.
    # For a square print (aspect ratio = 1): bbox_dist = dist / 4
    # This is what create_map_poster.py uses internally for all data fetching
    # and visual cropping, so --distance values are directly comparable.
    bbox_dist = dist / 4.0

    print("\n  Fetching map data...")
    g = fetch_graph(point, bbox_dist)
    if g is None:
        raise RuntimeError("Failed to retrieve street network. Cannot continue.")

    water_gdf = fetch_features(
        point, bbox_dist,
        tags={"natural": ["water", "bay", "strait"], "waterway": "riverbank"},
        name="water",
    )
    parks_gdf = fetch_features(
        point, bbox_dist,
        tags={"leisure": "park", "landuse": "grass"},
        name="parks",
    )

    print("\n  Processing geometry...")

    # Project graph and get CRS
    g_proj = ox.project_graph(g)
    target_crs = g_proj.graph["crs"]

    # Center point and bounding box in projected meters
    center_proj = project_center(lat, lon, target_crs)
    bbox = compute_bbox(center_proj, bbox_dist)
    bbox_geom = box(*bbox)

    # Extract layer geometries
    print("  Extracting water...")
    water_geom = extract_polygon_geometry(water_gdf, bbox_geom, target_crs)

    print("  Extracting parks...")
    parks_geom = extract_polygon_geometry(parks_gdf, bbox_geom, target_crs)

    print("  Extracting minor roads (may take a moment)...")
    roads_minor_geom = extract_road_geometry(g_proj, ROADS_MINOR_TYPES, BUFFER_MINOR_M, bbox_geom)

    print("  Extracting major roads (may take a moment)...")
    roads_major_geom = extract_road_geometry(g_proj, ROADS_MAJOR_TYPES, BUFFER_MAJOR_M, bbox_geom)

    # Coordinate transform for SVG
    transform_params = build_coordinate_transform(bbox, print_size_mm)

    # Marker position in OpenSCAD coordinates (cartesian, no Y flip)
    marker_pos = None
    if marker_lat is not None and marker_lon is not None:
        marker_pos = compute_openscad_position(marker_lat, marker_lon, bbox, print_size_mm, target_crs)
        if not (0 <= marker_pos[0] <= print_size_mm and 0 <= marker_pos[1] <= print_size_mm):
            print(f"    Warning: marker coordinates fall outside the map bbox — it won't be visible in the print")

    # Resolve display labels (fall back to city/country if not specified)
    title_city    = display_city    if display_city    else city
    title_country = display_country if display_country else country

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_slug = city.lower().replace(" ", "_")
    out_dir = Path(STL_OUTPUT_DIR) / f"{city_slug}_{theme_name}_{timestamp}"
    layers_dir = out_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    # Build layer configs
    layer_configs = build_layer_configs(theme, base_height_mm, layer_step_mm)

    # Load fonts early — needed for both title geometry and title SVG.
    font_files = load_fonts()

    # Convert the title text to shapely geometry in projected coordinates.
    # This lets us accumulate the actual letter outlines into the geo layers
    # below the title, giving the raised letters a solid base to sit on.
    print("  Computing title geometry for layer support...")
    title_geom = _clean_geom(_generate_title_geometry(
        title_city, title_country, font_files, bbox, print_size_mm
    ))

    # Accumulate geometries from ABOVE for physical support in multi-color FDM printing.
    # Each raised layer must cover the footprint of every layer ABOVE it so those
    # higher features have solid material beneath them and don't float in mid-air.
    #
    #   Layer 2 (water color)      = water ∪ parks ∪ minor_roads ∪ major_roads ∪ title
    #   Layer 3 (parks color)      =         parks ∪ minor_roads ∪ major_roads ∪ title
    #   Layer 4 (minor roads color)=                  minor_roads ∪ major_roads ∪ title
    #   Layer 5 (major roads color)=                               major_roads ∪ title
    #   Layer 6 (title color)      = title SVG only (no accumulation needed)
    def _union(*geoms):
        parts = [g for g in geoms if g is not None and not getattr(g, "is_empty", False)]
        return unary_union(parts) if parts else None

    roads_major_layer = _union(roads_major_geom, title_geom)
    roads_minor_layer = _union(roads_minor_geom, roads_major_geom, title_geom)
    parks_layer       = _union(parks_geom, roads_minor_geom, roads_major_geom, title_geom)
    water_layer       = _union(water_geom, parks_geom, roads_minor_geom, roads_major_geom, title_geom)

    # Clean all geometries before writing to SVG.
    # Removes self-intersecting rings, near-duplicate vertices, and sub-mm² fragments
    # that cause OpenSCAD F6 CGAL errors ("mesh is not closed").
    print("  Cleaning geometry (removing slivers and self-intersections)...")
    geom_map = {
        "background":  None,                             # handled as rect, no cleaning needed
        "water":       _clean_geom(water_layer),
        "parks":       _clean_geom(parks_layer),
        "roads_minor": _clean_geom(roads_minor_layer),
        "roads_major": _clean_geom(roads_major_layer),
    }

    # Write SVG files
    print("\n  Writing SVG layers...")
    for cfg in layer_configs:
        svg_path = layers_dir / cfg["filename"]
        geom = geom_map[cfg["name"]]
        write_layer_svg(
            svg_path, geom, transform_params, print_size_mm,
            cfg["color"], cfg["name"], cfg["is_background"],
        )
        print(f"    {cfg['filename']}")

    # Title SVG — rendered by matplotlib (text as bezier paths, same as poster)
    generate_title_svg(
        layers_dir / "06_title.svg",
        title_city, title_country, theme, print_size_mm, font_files,
    )
    print("    06_title.svg")

    # Write OpenSCAD file
    scad_name = f"map_{city_slug}.scad"
    scad_path = out_dir / scad_name
    print(f"\n  Writing {scad_name}...")
    generate_scad_file(
        scad_path, layer_configs, print_size_mm, base_height_mm, layer_step_mm,
        city, country, theme_name, lat, lon, dist, timestamp,
        display_city=title_city, display_country=title_country,
        marker_pos=marker_pos, marker_text=marker_text,
        marker_text_position=marker_text_position,
        title_color=title_color, marker_color=marker_color,
    )

    # Summary
    z_title  = base_height_mm + 4 * layer_step_mm
    z_marker = base_height_mm + 5 * layer_step_mm
    total_height = z_marker + layer_step_mm if marker_pos else z_title + layer_step_mm
    title_hex = layer_configs[-1]["color"]

    print("\n" + "=" * 60)
    print("  STL layer generation complete!")
    print("=" * 60)
    print(f"  Output folder : {out_dir}")
    print(f"  OpenSCAD file : {scad_path.name}")
    print(f"  Print size    : {print_size_mm}mm x {print_size_mm}mm")
    print(f"  Total height  : {total_height:.1f}mm")
    print()
    print("  Filament change heights:")
    for cfg in layer_configs:
        if cfg["index"] > 1:
            print(f"    {cfg['z_start']:.2f}mm  →  {cfg['color']}  ({cfg['color_label']})")
    print(f"    {z_title:.2f}mm  →  {title_color}  (title text)")
    if marker_pos:
        print(f"    {z_marker:.2f}mm  →  {marker_color}  (you-are-here marker)")
    print()
    print("  Next steps:")
    print("    1. Open the .scad file in OpenSCAD")
    print("    2. Press F5 to preview, F6 to render")
    print("    3. File → Export → Export as STL")
    print("    4. Import STL into your slicer and add color changes at the Z heights above")
    print("=" * 60)

    return out_dir


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D-printable map SVG layers and OpenSCAD file for any city.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_map_stl.py --city "Williamsville" --country "USA" --theme ocean
  python create_map_stl.py --city "Paris" --country "France" --distance 8000
  python create_map_stl.py --city "UB" --country "USA" --latitude 43.0018 --longitude -78.7830 --distance 5000
  python create_map_stl.py --list-themes
        """,
    )

    parser.add_argument("--city", "-c", type=str, help="City name")
    parser.add_argument("--country", "-C", type=str, help="Country name")
    parser.add_argument("--latitude", "-lat", dest="latitude", type=str,
                        help="Override latitude (decimal or DMS)")
    parser.add_argument("--longitude", "-long", dest="longitude", type=str,
                        help="Override longitude (decimal or DMS)")
    parser.add_argument("--distance", "-d", type=int, default=6000,
                        help="Map half-width in meters from center (default: 6000)")
    parser.add_argument("--theme", "-t", type=str, default="terracotta",
                        help="Theme name (default: terracotta)")
    parser.add_argument("--print-size", type=float, default=200.0,
                        help="Square print dimension in mm (default: 200)")
    parser.add_argument("--base-height", type=float, default=1.6,
                        help="Background plate height in mm (default: 1.6)")
    parser.add_argument("--layer-step", type=float, default=0.4,
                        help="Height per feature layer in mm (default: 0.4)")
    parser.add_argument("--display-city", "-dc", dest="display_city", type=str, default="",
                        help="Override city label on the poster (e.g. for non-latin scripts)")
    parser.add_argument("--display-country", "-dC", dest="display_country", type=str, default="",
                        help="Override country/subtitle label on the poster")
    parser.add_argument("--marker-lat", dest="marker_lat", type=str, default=None,
                        help="Latitude for 'You are here' star marker (decimal or DMS)")
    parser.add_argument("--marker-lon", dest="marker_lon", type=str, default=None,
                        help="Longitude for 'You are here' star marker (decimal or DMS)")
    parser.add_argument("--marker-text", dest="marker_text", type=str, default="You are here",
                        help="Label shown next to the star marker (default: 'You are here')")
    parser.add_argument("--marker-text-position", dest="marker_text_position", type=str,
                        default="right", choices=["right", "left", "above", "below"],
                        help="Position of the marker label relative to the star (default: right)")
    parser.add_argument("--title-color", dest="title_color", type=str, default="#1A1A1A",
                        help="Filament color hex for the title text layer (default: #1A1A1A)")
    parser.add_argument("--marker-color", dest="marker_color", type=str, default="#CC0000",
                        help="Filament color hex for the marker layer (default: #CC0000 red)")
    parser.add_argument("--list-themes", action="store_true",
                        help="List available themes and exit")

    args = parser.parse_args()

    print("=" * 60)
    print("  Map STL Layer Generator")
    print("=" * 60)

    if args.list_themes:
        themes = get_available_themes()
        if themes:
            print("\nAvailable themes:")
            for t in themes:
                print(f"  {t}")
        else:
            print(f"No themes found in '{THEMES_DIR}/' directory.")
        return

    if not args.city or not args.country:
        parser.error("--city and --country are required (unless using --list-themes)")

    # Load theme
    theme = load_theme(args.theme)

    # Resolve coordinates
    if args.latitude and args.longitude:
        lat = float(parse_latlon(args.latitude))
        lon = float(parse_latlon(args.longitude))
        print(f"  Coordinates: {lat}, {lon}")
        point = (lat, lon)
    else:
        point = get_coordinates(args.city, args.country)

    # Run
    try:
        # Parse optional marker coordinates
        marker_lat = float(parse_latlon(args.marker_lat)) if args.marker_lat else None
        marker_lon = float(parse_latlon(args.marker_lon)) if args.marker_lon else None
        if bool(args.marker_lat) != bool(args.marker_lon):
            print("  Warning: --marker-lat and --marker-lon must both be provided. Marker skipped.")
            marker_lat = marker_lon = None

        create_map_stl(
            city=args.city,
            country=args.country,
            point=point,
            dist=float(args.distance),
            theme_name=args.theme,
            theme=theme,
            print_size_mm=args.print_size,
            base_height_mm=args.base_height,
            layer_step_mm=args.layer_step,
            display_city=args.display_city,
            display_country=args.display_country,
            marker_lat=marker_lat,
            marker_lon=marker_lon,
            marker_text=args.marker_text,
            marker_text_position=args.marker_text_position,
            title_color=args.title_color,
            marker_color=args.marker_color,
        )
    except RuntimeError as e:
        print(f"\n  Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
