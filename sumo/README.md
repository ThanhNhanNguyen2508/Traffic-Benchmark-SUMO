# SUMO scenarios

```
sumo/
├── map/hcmc.net.xml.gz           study network (6 junctions), built by build_network.py
├── map/osm.net.xml.gz            original OSM import (geometry source only)
├── map/osm.poly.xml.gz           buildings / land use for sumo-gui
├── config/junctions.json         one entry per agent (J4 disabled: fixed time)
├── config/camera_segments.json   camera -> observed edges
├── bicycle/<mode>.bicycle.trips.xml   bicycles (not observed by cameras)
└── scenarios/
    ├── train/{light,medium,heavy}/   seed 1001
    ├── val/{light,medium,heavy}/     seed 2002
    ├── test/{light,medium,heavy}/    seed 3003
    └── stress/test-heavy/            original random 12x density stress test
```

Each scenario folder contains:

| File | Content |
|---|---|
| `osm.sumocfg` | open this in sumo-gui, or pass it to `sumo -c` |
| `osm.{motorcycle,passenger,truck,bus}.trips.xml` | vehicles with explicit routes (`<vehicle>` + `<route edges>`) |
| `osm.bicycle.trips.xml` | the original random bicycle trips of that setting |
| `counts.<class>.xml` | the per-edge targets the demand was sampled to |
| `scenario.json` | split, seed, scale, source hour, vehicle counts |

## How the demand is built (`build_demand.py`)

### Step 1: candidate routes

Every vehicle **enters at one of the 9 fringe edges and leaves at another**; nobody appears or disappears
mid-road.

| Fringe | Junction |
|---|---|
| Ly Thuong Kiet north / south | J0 / J2 |
| Bac Hai east | J5 |
| Thanh Thai south | J3 |
| To Hien Thanh east, Dong Nai | J4 |
| Lu Gia | J2 |
| Duong So 2 | J1 |
| Hem 319A | J0 |

`randomTrips --fringe-factor max` draws 36,000 fringe-to-fringe trips per class, and duarouter gives each the
shortest path that respects turn bans and vClass permissions. After removing duplicates, **73 distinct routes per
class** remain, one per reachable origin–destination pair. 63 of them pass at least one observed edge. The 10
that pass none (e.g. Lu Gia → Ly Thuong Kiet south, through J2 only) are never selected. No route ends in Hem 319A:
straight from Bac Hai and left from Ly Thuong Kiet south into the alley are banned at J0.

### Step 2: targets

For every camera, class and direction: `camera count ÷ 2 × scale` vehicles entering each observed edge in
[0, 3600) s. Example (heavy, motorcycles, scale 0.25):

| Observed edge | Target per direction |
|---|---:|
| Bac Hai J0↔J5 | 1,408 |
| Thanh Thai J5↔J3 | 1,414 |
| Ly Thuong Kiet J0↔J1 and J1↔J2 | 1,051 |
| To Hien Thanh J2↔J3 | 734 |

### Step 3: routeSampler

`routeSampler.py` repeatedly picks an observed edge that is still below target, draws at random a candidate route
through it that would not push any other observed edge over its target, adds one vehicle on that route, and
credits **every** observed edge the route passes. It stops when all targets are met; all observed edges match
100% (GEH < 5). A vehicle crossing two observed segments counts for both, so heavy needs only ~5,600 motorcycles
for ~11,300 counted passages. Departures are uniform in [0, 3600) s; vehicles insert on the best lane at maximum
speed.

Most used routes, test/heavy, motorcycles:

| Vehicles | Route | Observed edges passed |
|---:|---|---|
| 271 | Dong Nai → J4 → J3 → J5 → J0 → Ly Thuong Kiet north | Thanh Thai, Bac Hai |
| 260 | To Hien Thanh east → J4 → J3 → J5 → J0 → Ly Thuong Kiet north | Thanh Thai, Bac Hai |
| 255 | Thanh Thai south → J3 → J5 → J0 → Ly Thuong Kiet north | Thanh Thai, Bac Hai |
| 214 | Dong Nai → J4 → J3 → J5 → Bac Hai east | Thanh Thai |

First vehicle of `test/heavy/osm.motorcycle.trips.xml`: at 0.16 s, a motorcycle enters from Ly Thuong Kiet south,
drives To Hien Thanh to J3 and leaves by Thanh Thai south:

```xml
<vehicle id="motorcycle_0" depart="0.16" type="motorcycle_motorcycle" departLane="best" departSpeed="max">
    <route edges="LTK_B_LTK_S_J2 THT_J2_J3 TT_J3_B_TT_S"/>
</vehicle>
```

The cameras only give totals per segment. **Turning ratios and origin–destination shares are not measured**;
they come from the random draw. The **seed of the split** drives that draw and the departure times. Therefore
train, val and test have identical flow on every observed edge but different routes and timings. Unobserved
roads (Lu Gia, Dong Nai, Duong So 2, Hem 319A) carry whatever the chosen routes imply.

### Differences from the original random demand (`build_trips.py`)

| | original (`build_trips.py`, stress test) | benchmark (`build_demand.py`) |
|---|---|---|
| amount | insertion density, vehicles/km/h (1×, 2×, 3×, 12×) | camera counts × 0.25 |
| origins / destinations | 80% / 75% at fringe edges, the rest mid-network | 100% fringe |
| flow on a road | not controlled | matches the camera on observed edges |
| departures | evenly spaced (e.g. one motorcycle every 2.31 s) | uniform random |
| file | `<trip from to>` (routed at run time) | `<vehicle>` with its route |

## Scale

At 100% of the camera flow, light reaches only 45–59% of its targets and ~5,100 vehicles are still waiting to enter
after one hour (medium 32–49%, heavy 28–37%). A segment carries ~3,500–4,500 veh/h in SUMO. With **scale 0.25**
(1-hour run, placeholder fixed-time plan):

| Setting | Vehicles | Waiting to enter at the end | Mean time loss | Reached target |
|---|---:|---:|---:|---:|
| light | 3,936 | 0 | 89 s | 97–99% |
| medium | 5,126 | 1 | 137 s | 97–99% |
| heavy | 6,845 | 28 | 161 s | 90–96% |

(measured with plain `sumo -c` on an earlier seed; the split files have the same targets).
