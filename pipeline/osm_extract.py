#!/usr/bin/env python3
"""
pipeline/osm_extract.py — Parse OSM data into a compact SQLite road graph
for use by the Android HMM map-matcher.

Usage:
    # From an .osm.xml file (download from https://www.openstreetmap.org/export)
    python3 pipeline/osm_extract.py --input datasets/area.osm --output datasets/road_graph.db

    # From a .osm.pbf file (Geofabrik regional extracts)
    python3 pipeline/osm_extract.py --input datasets/region.osm.pbf --output datasets/road_graph.db

Output SQLite schema
--------------------
nodes(id INTEGER PRIMARY KEY, lat REAL NOT NULL, lon REAL NOT NULL)
edges(id       INTEGER PRIMARY KEY,
      from_id  INTEGER NOT NULL REFERENCES nodes(id),
      to_id    INTEGER NOT NULL REFERENCES nodes(id),
      lat1 REAL, lon1 REAL, lat2 REAL, lon2 REAL,
      bearing_rad REAL,   -- forward bearing from→to (radians)
      length_m    REAL,   -- great-circle length (metres)
      highway     TEXT)   -- OSM highway tag value

Indexes:
  idx_edges_from   (from_id)
  idx_edges_latbox (lat1, lon1, lat2, lon2)  -- bounding-box spatial lookup

Requires: pip install osmium numpy
"""

import argparse
import math
import os
import sqlite3
import sys
from pathlib import Path

try:
    import osmium
    OSMIUM_AVAILABLE = True
except ImportError:
    OSMIUM_AVAILABLE = False

# ─── OSM highway types we care about (exclude footways, etc.) ────────────────

DRIVEABLE = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "unclassified", "residential",
    "service", "living_street", "road",
}

# ─── Geometry helpers ─────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres."""
    R = 6_378_137.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def bearing(lat1, lon1, lat2, lon2):
    """Forward bearing from (lat1,lon1) → (lat2,lon2) in radians [-π, π]."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2)
         - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return math.atan2(x, y)

# ─── Fallback XML parser (no osmium) ─────────────────────────────────────────

def parse_xml_fallback(osm_path: Path):
    """
    Very lightweight OSM XML parser using ElementTree.
    Handles .osm.xml files only (not PBF). Returns (nodes_dict, ways_list).
    nodes_dict: {node_id: (lat, lon)}
    ways_list:  [(node_id_list, highway_tag)]
    """
    import xml.etree.ElementTree as ET
    print("  Using ElementTree XML parser (install osmium for PBF + speed)")
    nodes = {}
    ways = []
    context = ET.iterparse(str(osm_path), events=("start",))
    current_way_nodes = []
    current_highway = None
    in_way = False
    for event, elem in context:
        if elem.tag == "node":
            nid = int(elem.get("id"))
            lat = float(elem.get("lat", 0))
            lon = float(elem.get("lon", 0))
            nodes[nid] = (lat, lon)
        elif elem.tag == "way":
            in_way = True
            current_way_nodes = []
            current_highway = None
        elif in_way and elem.tag == "nd":
            current_way_nodes.append(int(elem.get("ref")))
        elif in_way and elem.tag == "tag":
            if elem.get("k") == "highway":
                current_highway = elem.get("v")
        elif in_way and elem.tag == "way" and event == "start":
            pass
        # End of way is tricky with iterparse start-only; use a different approach
        elem.clear()

    # Re-parse for ways properly
    tree = ET.parse(str(osm_path))
    root = tree.getroot()
    for way in root.findall("way"):
        hw = None
        for tag in way.findall("tag"):
            if tag.get("k") == "highway":
                hw = tag.get("v")
        if hw not in DRIVEABLE:
            continue
        nds = [int(nd.get("ref")) for nd in way.findall("nd")]
        if len(nds) >= 2:
            ways.append((nds, hw))
    return nodes, ways


# ─── osmium handler ───────────────────────────────────────────────────────────

class RoadHandler(osmium.SimpleHandler if OSMIUM_AVAILABLE else object):  # type: ignore
    def __init__(self):
        if OSMIUM_AVAILABLE:
            super().__init__()
        self.nodes = {}    # id → (lat, lon)
        self.ways  = []    # [(node_id_list, highway_tag)]

    def node(self, n):
        self.nodes[n.id] = (n.location.lat, n.location.lon)

    def way(self, w):
        hw = w.tags.get("highway")
        if hw not in DRIVEABLE:
            return
        refs = [n.ref for n in w.nodes]
        if len(refs) >= 2:
            self.ways.append((refs, hw))


# ─── Build SQLite DB ──────────────────────────────────────────────────────────

def build_db(nodes: dict, ways: list, db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE nodes (
            id  INTEGER PRIMARY KEY,
            lat REAL NOT NULL,
            lon REAL NOT NULL
        );
        CREATE TABLE edges (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            from_id     INTEGER NOT NULL,
            to_id       INTEGER NOT NULL,
            lat1        REAL NOT NULL,
            lon1        REAL NOT NULL,
            lat2        REAL NOT NULL,
            lon2        REAL NOT NULL,
            bearing_rad REAL NOT NULL,
            length_m    REAL NOT NULL,
            highway     TEXT
        );
    """)

    # Insert nodes (only those referenced by driveable ways)
    referenced = {ref for way_nodes, _ in ways for ref in way_nodes}
    node_rows = [(nid, lat, lon)
                 for nid, (lat, lon) in nodes.items()
                 if nid in referenced]
    cur.executemany("INSERT OR IGNORE INTO nodes VALUES (?,?,?)", node_rows)

    # Insert directed edges (one per consecutive node pair, both directions)
    edge_rows = []
    for way_nodes, hw in ways:
        for i in range(len(way_nodes) - 1):
            a, b = way_nodes[i], way_nodes[i + 1]
            if a not in nodes or b not in nodes:
                continue
            lat1, lon1 = nodes[a]
            lat2, lon2 = nodes[b]
            dist = haversine(lat1, lon1, lat2, lon2)
            brg  = bearing(lat1, lon1, lat2, lon2)
            edge_rows.append((a, b, lat1, lon1, lat2, lon2, brg,  dist, hw))
            # Reverse direction
            edge_rows.append((b, a, lat2, lon2, lat1, lon1,
                               brg + math.pi if brg < 0 else brg - math.pi,
                               dist, hw))

    cur.executemany(
        "INSERT INTO edges(from_id,to_id,lat1,lon1,lat2,lon2,bearing_rad,length_m,highway)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        edge_rows)

    # Spatial + graph indexes
    cur.executescript("""
        CREATE INDEX idx_edges_from   ON edges(from_id);
        CREATE INDEX idx_edges_latbox ON edges(lat1, lon1, lat2, lon2);
        CREATE INDEX idx_nodes_latlon ON nodes(lat, lon);
    """)
    con.commit()
    con.close()

    return len(node_rows), len(edge_rows)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Build SQLite road graph from OSM data")
    ap.add_argument("--input",  required=True, help=".osm.xml or .osm.pbf file")
    ap.add_argument("--output", default="datasets/road_graph.db")
    ap.add_argument("--bbox",   nargs=4, type=float, metavar=("MIN_LAT","MIN_LON","MAX_LAT","MAX_LON"),
                    help="Bounding box filter (optional, keeps only nodes inside)")
    args = ap.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"ERROR: Input file not found: {in_path}")
        sys.exit(1)

    print(f"Parsing OSM: {in_path} ...")

    if OSMIUM_AVAILABLE and in_path.suffix in (".pbf", ".osm"):
        handler = RoadHandler()
        handler.apply_file(str(in_path), locations=True)
        nodes, ways = handler.nodes, handler.ways
    else:
        nodes, ways = parse_xml_fallback(in_path)

    print(f"  Raw nodes: {len(nodes):,}  ways: {len(ways):,}")

    # Optional bounding box filter
    if args.bbox:
        min_lat, min_lon, max_lat, max_lon = args.bbox
        nodes = {nid: (lat, lon) for nid, (lat, lon) in nodes.items()
                 if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon}
        ways = [(nds, hw) for nds, hw in ways
                if any(nid in nodes for nid in nds)]
        print(f"  After bbox filter: {len(nodes):,} nodes  {len(ways):,} ways")

    print(f"Building SQLite road graph: {out_path} ...")
    n_nodes, n_edges = build_db(nodes, ways, out_path)

    size_kb = out_path.stat().st_size / 1024
    print(f"  Nodes: {n_nodes:,}  Edges: {n_edges:,}  Size: {size_kb:.0f} KB")
    print(f"Done → {out_path}")


if __name__ == "__main__":
    main()
