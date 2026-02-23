# Map-to-STL: Multi-Color FDM 3D Printing from OpenStreetMap Data

## Project Overview

`create_map_stl.py` is a companion script to [`create_map_poster.py`](create_map_poster.py) from the [maptoposter](https://github.com/originalankur/maptoposter) project. The original tool generates beautiful PNG map posters from OpenStreetMap geodata. This extension takes that same underlying OSM vector data and produces layered STL-ready files for multi-color FDM 3D printing — working directly with the geodata rather than the rendered image. This gives precise control over road widths, layer heights, and spatial geometry before anything reaches a slicer.

### Output

```
stl_output/
└── {city}_{theme}_{timestamp}/
    ├── layers/
    │   ├── 01_background.svg
    │   ├── 02_water.svg
    │   ├── 03_parks.svg
    │   ├── 04_roads_minor.svg
    │   ├── 05_roads_major.svg
    │   └── 06_title.svg
    └── map_{city}.scad          ← open in OpenSCAD, F5 preview, F6 render, export STL
```

### Filament Color Changes (default heights)

| Z (mm) | Layer | Color (terracotta theme) |
|--------|-------|--------------------------|
| 0 – 1.6 | Background | Background color |
| 1.6 – 2.0 | Water | Water color |
| 2.0 – 2.4 | Parks | Park/green color |
| 2.4 – 2.8 | Minor roads | Tertiary road color |
| 2.8 – 3.2 | Major roads | Primary road color |
| 3.2 – 3.6 | Title text | Accent/contrast color |

Add M600 (filament change) pauses in your slicer at each Z boundary.

---

## Architecture

### Working from Vector Data

Rather than starting from the rendered PNG output, this script taps into the same OSM data pipeline that maptoposter already uses, processing it through the same cache and theme system but directing the output toward SVG layers for OpenSCAD instead of a PNG poster. This approach was chosen after exploring image-based routes — working from raster data presents challenges with resolution, color segmentation, and clean geometry that are straightforward to sidestep entirely by staying in vector space.

### Shared Cache

Both scripts use identical cache keys (e.g., `graph_43.0_-78.0_5000`). Running `create_map_stl.py` after `create_map_poster.py` for the same city/distance will print "Using cached" for all OSM data — no duplicate downloads.

### Coordinate System Pipeline

```
OSM (EPSG:4326, lat/lon degrees)
    → project via osmnx to local UTM (meters, cartesian)
    → clip to square bounding box (center ± dist meters)
    → affine transform to SVG mm space (Y-flipped, 1 unit = 1 mm)
    → OpenSCAD reads SVG mm directly (no scale() needed)
```

The affine transform parameters:

```
scale = print_size_mm / (dist * 2)
[a, b, d, e, xoff, yoff] = [scale, 0, 0, -scale, -minx*scale, maxy*scale]
```

This maps `(minx, maxy)` → `(0, 0)` (top-left) and `(maxx, miny)` → `(print_size, print_size)` (bottom-right), with Y flipped to match SVG's top-down axis.

---

## Layer Design

### Physical Constraint: Every Raised Layer Needs Support Beneath It

In single-material FDM, support is automatic. In multi-color FDM without a wipe tower, each filament color forms an entirely separate region of the print at its Z range. **A raised feature has no material beneath it unless the layer below explicitly covers its footprint.**

This means each layer must include the union of all geometry from all layers above it:

```
Layer 2 (water):       water ∪ parks ∪ minor_roads ∪ major_roads ∪ title
Layer 3 (parks):       parks ∪ minor_roads ∪ major_roads ∪ title
Layer 4 (minor roads): minor_roads ∪ major_roads ∪ title
Layer 5 (major roads): major_roads ∪ title
Layer 6 (title):       title only (topmost — nothing above it)
```

Layer 1 (background) is always a full square, so it needs no modification.

This ensures every feature that appears at any height has solid geometry supporting it all the way from the background up.

---

## Title Text Layer

### Challenge: OpenSCAD SVG Limitations

OpenSCAD's `import()` for SVG files has significant restrictions that aren't documented clearly:

1. **CSS `style` attributes are ignored.** `<path style="fill:#ff0000">` renders as black. Only inline attributes work: `<path fill="#ff0000">`.
2. **`<g transform="matrix(...)">` is ignored.** Group-level transforms are not applied. This affects standard matplotlib SVG output.
3. **Only `<path>` and `<rect>` elements are rendered.** Text elements are silently skipped.

### matplotlib Figure Export and OpenSCAD

When you do `fig.savefig('title.svg', format='svg')` with `matplotlib.rcParams['svg.fonttype'] = 'path'`, matplotlib:

- Converts each glyph to bezier paths in raw font-unit coordinates (e.g., x=3850, y=4550)
- Wraps them in `<g transform="matrix(scale, 0, 0, -scale, tx, ty)">` to position them
- OpenSCAD ignores the transform → glyphs render at raw font-unit coordinates, far outside the viewBox → invisible

### Solution: `matplotlib.textpath.TextPath`

`TextPath` generates glyph bezier vertices directly in the coordinate space you specify — no figure, no axes, no transforms:

```python
tp = TextPath((x_origin, y_origin), text, prop=font_properties)
# tp.vertices are already in pt-space at the given origin
```

With a `ViewBox="0 0 864 864"` (864pt = 12in × 72pt/in, matching the poster's `figsize=12`):

- All glyph coordinates land in `[0, 864]` range
- SVG `width="200mm" height="200mm"` maps the viewBox to physical mm
- OpenSCAD reads the file: 1 SVG unit = 1 mm (no scale correction needed)

### Custom SVG Path Code Conversion

`TextPath` produces a `matplotlib.path.Path` with vertices and codes. The codes must be manually converted to SVG `d` attribute commands, including Y-axis flip (matplotlib Y-up, SVG Y-down):

```python
svg_y = viewbox_height - mpl_y  # flip

MOVETO    → M x y
LINETO    → L x y
CURVE3    → Q cx cy ex ey     (quadratic bezier: 2 verts consumed)
CURVE4    → C c1x c1y c2x c2y ex ey  (cubic: 3 verts consumed)
CLOSEPOLY → Z
```

`fill-rule="evenodd"` on the path element handles holes correctly (letter counters like the inside of 'O').

### Title Geometry for Layer Accumulation

The title SVG is only used by OpenSCAD for the topmost layer. For the lower layers (parks, roads) to physically support the title text, we need the title letter shapes as **shapely geometry** so they can be unioned into each layer's SVG.

`TextPath.to_polygons()` approximates all bezier curves as polylines and splits on each MOVETO, returning one numpy array per contour component (outer rings and holes mixed together).

#### TrueType Winding Order

The TrueType font specification defines:
- **Outer contours (filled areas): clockwise (CW)**
- **Inner counters/holes (e.g., interior of 'O', 'A', 'B', 'P', 'R'): counter-clockwise (CCW)**

This is opposite to the PDF/PostScript convention (where outer = CCW). The shoelace formula gives:
- CW ring → negative signed area
- CCW ring → positive signed area

The correct classification:
```python
signed_area = shoelace(polygon_points)
if signed_area <= 0:
    shapes_outer.append(polygon)   # CW = outer ring
else:
    shapes_holes.append(polygon)   # CCW = hole/counter
```

After separating: `result = unary_union(outer_rings).difference(unary_union(holes))`

---

## OpenSCAD Integration

### Generated `.scad` Structure

```openscad
// Filament change guide:
//   Pause at Z=1.6mm  → swap to water color
//   Pause at Z=2.0mm  → swap to parks color
//   ...

print_size  = 200;
base_height = 1.6;
layer_step  = 0.4;

module map_layer(svg_file, height) {
    linear_extrude(height) { import(svg_file); }
}

module layer_background() {
    color("#c8a882") translate([0,0,0])
        map_layer("layers/01_background.svg", base_height);
}
// ... (one module per layer)

union() {
    layer_background();
    layer_water();
    layer_parks();
    layer_roads_minor();
    layer_roads_major();
    layer_title();
}
```

### No `scale()` Needed

Because the SVG viewBox is in mm units (`width="200mm" viewBox="0 0 200 200"`), OpenSCAD's `import()` reads it at 1:1 — 1 SVG unit = 1 mm. No `scale()` wrapper is required in the `.scad` file.

---

## CLI Usage

```bash
python create_map_stl.py \
    --city "Buffalo" \
    --country "USA" \
    --display-city "University at Buffalo" \
    --display-country "North Campus" \
    --latitude 43.0019 \
    --longitude -78.7877 \
    --distance 6000 \
    --theme ocean \
    --print-size 200 \
    --base-height 1.6 \
    --layer-step 0.4
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--city` / `-c` | required | City name (used for geocoding and output folder name) |
| `--country` / `-C` | required | Country name (used for geocoding) |
| `--latitude` / `-lat` | (geocoded) | Override center latitude |
| `--longitude` / `-long` | (geocoded) | Override center longitude |
| `--distance` / `-d` | 6000 | Half-width of bounding box in meters |
| `--theme` / `-t` | terracotta | Theme name (matches `themes/*.yaml`) |
| `--print-size` | 200.0 | Output square dimension in mm |
| `--base-height` | 1.6 | Background layer height in mm (4 × 0.4mm layers) |
| `--layer-step` | 0.4 | Height per raised layer in mm (1 print layer) |
| `--display-city` / `-dc` | (same as `--city`) | Override the city label printed on the map (e.g. "University at Buffalo") |
| `--display-country` / `-dC` | (same as `--country`) | Override the subtitle label (e.g. "North Campus") |
| `--list-themes` | — | Print available themes and exit |

---

## Dependencies

All dependencies are already installed in the maptoposter virtual environment:

| Package | Used for |
|---------|----------|
| `osmnx` | OSM graph + feature download, UTM projection |
| `geopandas` | GeoDataFrame projection and clipping |
| `shapely` | Geometry operations (buffer, union, difference, affine transform) |
| `matplotlib` | `TextPath` for font glyph bezier paths (no display, `Agg` backend) |
| `geopy` | Nominatim geocoding |
| `lat_lon_parser` | Flexible lat/lon string parsing |
| `pyyaml` | Theme file loading |

---

## Design Decisions

### Square Bounding Box (no aspect ratio compensation)

`create_map_poster.py` applies a compensation factor to the OSM fetch distance to account for the poster's aspect ratio. Since `create_map_stl.py` always produces a square print, `--distance` is used directly as the half-width in both X and Y: `bbox_dist = dist / 4.0` (matching the poster's `compensated_dist` formula for 1:1 aspect ratio).

### Road Buffering in Projected Space

Roads from OSM are `LineString` geometries. They're buffered in projected meter space (not geographic degrees) for accurate physical widths:
- Major roads (motorway, primary, secondary): 25m radius
- Minor roads (tertiary, residential, living street): 12m radius

### Water and Park Feature Extraction

OSM features are fetched with `ox.features_from_point()`. Only `Polygon` and `MultiPolygon` geometries are used (lines and points are discarded). All features are unioned into a single shapely geometry and clipped to the square bounding box.

### Theme Color Mapping

The script reads the same `themes/*.yaml` files as the poster script. Color keys used:
- `bg` — background
- `water` — water bodies
- `parks` (or `green`) — parks and green spaces
- `road_tertiary` — minor roads
- `road_primary` — major roads

Missing keys fall back to `#888888` with a warning.

---

## Mesh Geometry: OpenSCAD and Slicer Solutions

Two geometry problems were worked through during development.

### OpenSCAD F6 "Mesh Not Closed"

**Error:** `ERROR: The given mesh is not closed! Unable to convert to CGAL_Nef_Polyhedron.`

OpenSCAD's CGAL backend (used for F6 full render and STL export) is stricter than the F5 preview renderer. It requires all 2D polygons to produce a valid closed 3D mesh when extruded.

**Root causes identified:**
1. **Self-intersecting rings** — `unary_union` of road buffers at complex junctions produces T-junctions or rings that cross by a floating-point epsilon
2. **Near-duplicate vertices** — `TextPath.to_polygons()` bezier approximation produces vertices < 1µm apart that survive shapely but confuse CGAL
3. **Sub-millimeter fragments** — tiny sliver polygons (area < 0.5 m²) produce degenerate CGAL faces
4. **Shared edges in a single path** — writing all polygons into one `<path>` element with `fill-rule="evenodd"` causes adjacent polygons sharing an edge to cancel each other out, producing zero-width seams CGAL can't represent

**Solutions:**

`_clean_geom()` applied to all layer geometries before SVG write:
```python
geom = geom.buffer(0)                # repair self-intersecting rings
geom = geom.buffer(0.5).buffer(-0.5) # merge touching-at-a-point regions
geom = geom.simplify(0.5, preserve_topology=True)
# + filter fragments with area < 0.5 m²
```

One `<path>` element per `Polygon` instead of one combined path — eliminates cross-polygon evenodd cancellation at shared edges.

### Cura "Model is not watertight"

**Warning:** `Model is not watertight, and may not print properly.`

Even with CGAL passing, the exported STL had coincident internal faces at each Z boundary — the top of one layer at exactly the same Z as the bottom of the next. Some slicer mesh validators flag these as non-manifold seams.

**Solution:** Each layer in the `.scad` file overlaps the layer below by 0.01mm:

```
Background: Z = 0       → 1.61  (extends 0.01 into water zone)
Water:      Z = 1.59    → 2.01  (sinks 0.01 into background, extends 0.01 into parks)
Parks:      Z = 1.99    → 2.41  (sinks 0.01 into water, extends 0.01 into minor roads)
...
```

0.01mm is completely invisible at print scale. The color change Z heights in the slicer are set manually and are unaffected by this geometry overlap.

---

## Known Limitations

- **OpenSCAD F6 render** is slow for complex city geometries. F5 (preview) is faster for iteration.
- **Large distances** can produce SVG files >5 MB, which may slow OpenSCAD. Use `--distance 4000` or smaller if needed.
- **Highway attribute lists**: OSM sometimes returns `highway` as a Python list (multiple values). The script takes `[0]` with fallback to `'unclassified'`.
- **Missing edge geometry**: Some OSM graph edges lack a `geometry` attribute. The script reconstructs a `LineString` from the edge's source and target node coordinates.
- **Font availability**: The script loads Roboto fonts from `fonts/`. If a font file is missing, it falls back to matplotlib's default font. Glyph shapes may differ slightly.
