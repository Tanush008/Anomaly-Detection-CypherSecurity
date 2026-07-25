import random
import uuid
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
random.seed(42)
np.random.seed(42)

N_USERS = 40
N_SERVICE_ACCOUNTS = 10
N_EDGE_DEVICES = 10
DAYS = 30

RESOURCE_POOL = {
    "user": ["hr_portal", "finance_db", "code_repo", "customer_crm", "email_server", "admin_console"],
    "service_account": ["billing_api", "analytics_dash", "customer_crm", "finance_db"],
    "edge_device": ["vpn_gateway", "file_share", "sensor_gateway"],
}
AUTH_METHODS = {
    "user": ["password", "biometric"],
    "service_account": ["token", "certificate"],
    "edge_device": ["certificate"],
}
CITIES = [
    ("Indore", 22.7196, 75.8577), ("Mumbai", 19.0760, 72.8777),
    ("Delhi", 28.7041, 77.1025), ("Bengaluru", 12.9716, 77.5946),
    ("Hyderabad", 17.3850, 78.4867), ("Singapore", 1.3521, 103.8198),
    ("London", 51.5074, -0.1278), ("New York", 40.7128, -74.0060),
    ("Moscow", 55.7558, 37.6173), ("Lagos", 6.5244, 3.3792),
]
OS_FINGERPRINTS = ["Windows11-23H2", "macOS-14.5", "Ubuntu-22.04", "iOS-17.4", "Android-14", "FirmwareRTOS-3.2"]
ACTIONS = ["login", "list_dir", "read_file", "write_file", "download", "modify_permissions", "logout"]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def make_entities():
    entities = []
    for i in range(N_USERS):
        entities.append(_make_entity(f"user_{i:03d}", "user"))
    for i in range(N_SERVICE_ACCOUNTS):
        entities.append(_make_entity(f"svc_{i:03d}", "service_account"))
    for i in range(N_EDGE_DEVICES):
        entities.append(_make_entity(f"edge_{i:03d}", "edge_device"))
    return entities


def _make_entity(entity_id, entity_type):
    home_city = random.choice(CITIES)
    mac = ":".join(f"{random.randint(0,255):02X}" for _ in range(6))
    fingerprint = f"{random.choice(OS_FINGERPRINTS)}|{mac}"
    if entity_type == "user":
        hour_mean, hour_std = random.randint(8, 18), random.uniform(1.0, 2.0)
    elif entity_type == "service_account":
        hour_mean, hour_std = 12, 8.0  # effectively runs all day, wide spread = "always on"
    else:
        hour_mean, hour_std = random.choice([2, 8, 14, 20]), 1.5  # periodic check-in slot

    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "home_city": home_city,
        "hour_mean": hour_mean,
        "hour_std": hour_std,
        "usual_resources": random.sample(RESOURCE_POOL[entity_type], k=min(3, len(RESOURCE_POOL[entity_type]))),
        "device_id": f"dev_{uuid.uuid4().hex[:8]}",
        "device_fingerprint": fingerprint,
        "auth_method": random.choice(AUTH_METHODS[entity_type]),
        "privileged": entity_type == "user" and random.random() < 0.2,
    }


def command_sequence_for(resource, privileged, n_min=2, n_max=4):
    base = ["login"] + random.sample(ACTIONS[1:-1], k=random.randint(n_min, n_max))
    if privileged and resource == "admin_console":
        base.append("modify_permissions")
    base.append("logout")
    return "->".join(base)


def normal_row(entity, ts, resource=None):
    hour = int(np.clip(np.random.normal(entity["hour_mean"], entity["hour_std"]), 0, 23))
    ts = ts.replace(hour=hour, minute=random.randint(0, 59))
    city, lat, lon = entity["home_city"]
    resource = resource or random.choice(entity["usual_resources"])
    duration = max(1, np.random.normal(15 if entity["entity_type"] == "user" else 4, 6))
    return {
        "timestamp": ts, "entity_id": entity["entity_id"], "entity_type": entity["entity_type"],
        "device_id": entity["device_id"], "device_fingerprint": entity["device_fingerprint"],
        "source_ip": fake.ipv4_public(), "geo_location": city, "lat": lat + np.random.normal(0, 0.02),
        "lon": lon + np.random.normal(0, 0.02),
        "resource_accessed": resource, "auth_method": entity["auth_method"],
        "session_duration": round(duration, 1),
        "command_sequence": command_sequence_for(resource, entity["privileged"]),
        "login_success": True,
        "pattern": "normal_baseline", "signal_type": "Benign",
    }


def inject_brute_force(entity, ts, rows):
    base = ts.replace(hour=random.randint(1, 4), minute=random.randint(0, 30))
    city, lat, lon = entity["home_city"]
    for i in range(random.randint(10, 25)):
        rows.append({
            "timestamp": base + timedelta(seconds=i * random.randint(3, 8)),
            "entity_id": entity["entity_id"], "entity_type": entity["entity_type"],
            "device_id": entity["device_id"], "device_fingerprint": entity["device_fingerprint"],
            "source_ip": fake.ipv4_public(), "geo_location": city, "lat": lat, "lon": lon,
            "resource_accessed": "vpn_gateway", "auth_method": entity["auth_method"],
            "session_duration": 0.2, "command_sequence": "login",
            "login_success": False, "pattern": "brute_force", "signal_type": "Anomaly",
        })
    rows.append({**rows[-1], "timestamp": base + timedelta(minutes=1), "login_success": True,
                 "command_sequence": "login->read_file->logout", "session_duration": 3.0})


def inject_impossible_travel(entity, ts, rows):
    rows.append(normal_row(entity, ts))
    hlat, hlon = entity["home_city"][1], entity["home_city"][2]
    far_city, flat, flon = random.choice([c for c in CITIES if haversine_km(hlat, hlon, c[1], c[2]) > 2000])
    second = ts + timedelta(minutes=random.randint(5, 30))
    rows.append({
        "timestamp": second, "entity_id": entity["entity_id"], "entity_type": entity["entity_type"],
        "device_id": f"dev_{uuid.uuid4().hex[:8]}", "device_fingerprint": random.choice(OS_FINGERPRINTS),
        "source_ip": fake.ipv4_public(), "geo_location": far_city, "lat": flat, "lon": flon,
        "resource_accessed": "email_server", "auth_method": entity["auth_method"], "session_duration": 5.0,
        "command_sequence": "login->read_file->logout",
        "login_success": True, "pattern": "impossible_travel", "signal_type": "Anomaly",
    })


def inject_lateral_movement(entity, ts, rows):
    all_resources = sum(RESOURCE_POOL.values(), [])
    unused = [r for r in all_resources if r not in entity["usual_resources"]]
    city, lat, lon = entity["home_city"]
    for i, res in enumerate(random.sample(unused, k=min(4, len(unused)))):
        rows.append({
            "timestamp": ts + timedelta(minutes=i * 3), "entity_id": entity["entity_id"],
            "entity_type": entity["entity_type"], "device_id": entity["device_id"],
            "device_fingerprint": entity["device_fingerprint"], "source_ip": fake.ipv4_public(),
            "geo_location": city, "lat": lat, "lon": lon, "resource_accessed": res,
            "auth_method": entity["auth_method"], "session_duration": 8.0,
            "command_sequence": "login->list_dir->read_file->download->logout",
            "login_success": True, "pattern": "lateral_movement", "signal_type": "Anomaly",
        })


def inject_device_spoofing(entity, ts, rows):
    city, lat, lon = entity["home_city"]
    odd_ts = ts.replace(hour=random.choice([0, 1, 2, 3]), minute=random.randint(0, 59))
    rows.append({
        "timestamp": odd_ts, "entity_id": entity["entity_id"], "entity_type": entity["entity_type"],
        "device_id": entity["device_id"],  # SAME device_id...
        "device_fingerprint": random.choice([f for f in OS_FINGERPRINTS if f != entity["device_fingerprint"].split("|")[0]]) + "|" + ":".join(f"{random.randint(0,255):02X}" for _ in range(6)),  # ...mismatched fingerprint
        "source_ip": fake.ipv4_public(), "geo_location": city, "lat": lat, "lon": lon,
        "resource_accessed": "admin_console", "auth_method": entity["auth_method"], "session_duration": 6.0,
        "command_sequence": "login->modify_permissions->logout",
        "login_success": True, "pattern": "device_spoofing", "signal_type": "Anomaly",
    })


def inject_credential_stuffing(entities, ts, rows, n_targets=20):
    """Global pattern: a small set of attacker IPs hit MANY entity_ids with failed auth."""
    attacker_ips = [fake.ipv4_public() for _ in range(2)]
    targets = random.sample(entities, k=min(n_targets, len(entities)))
    base = ts.replace(hour=random.randint(1, 5), minute=0)
    for i, entity in enumerate(targets):
        city, lat, lon = entity["home_city"]
        rows.append({
            "timestamp": base + timedelta(seconds=i * random.randint(2, 6)),
            "entity_id": entity["entity_id"], "entity_type": entity["entity_type"],
            "device_id": f"dev_{uuid.uuid4().hex[:8]}", "device_fingerprint": random.choice(OS_FINGERPRINTS),
            "source_ip": random.choice(attacker_ips), "geo_location": city, "lat": lat, "lon": lon,
            "resource_accessed": "vpn_gateway", "auth_method": entity["auth_method"], "session_duration": 0.1,
            "command_sequence": "login", "login_success": (random.random() < 0.05),
            "pattern": "credential_stuffing", "signal_type": "Anomaly",
        })


def inject_low_and_slow_exfiltration(entity, start_ts, rows, n_days=10):
    """Gradual off-hours resource access building up over days/weeks — deliberately
    subtle at the single-event level so per-row rules mostly miss it; this is the
    scenario the SEQUENCE model earns its place by catching."""
    city, lat, lon = entity["home_city"]
    sensitive = "finance_db" if "finance_db" in RESOURCE_POOL[entity["entity_type"]] else entity["usual_resources"][0]
    for d in range(n_days):
        ts = start_ts + timedelta(days=d)
        ts = ts.replace(hour=random.choice([22, 23, 0, 1]), minute=random.randint(0, 59))
        duration = 5 + d * 2.5  # grows day over day
        rows.append({
            "timestamp": ts, "entity_id": entity["entity_id"], "entity_type": entity["entity_type"],
            "device_id": entity["device_id"], "device_fingerprint": entity["device_fingerprint"],
            "source_ip": fake.ipv4_public(), "geo_location": city, "lat": lat, "lon": lon,
            "resource_accessed": sensitive, "auth_method": entity["auth_method"],
            "session_duration": round(duration, 1),
            "command_sequence": "login->list_dir->read_file->download->logout",
            "login_success": True, "pattern": "low_and_slow_exfiltration", "signal_type": "Anomaly",
        })


def inject_insider_drift(entity, start_ts, rows, n_days=14):
    """Legitimate entity slowly expanding its own resource footprint — plausible,
    not clearly malicious. Labeled 'Edge case', not 'Anomaly' — used to tune the
    false-positive budget rather than to be reliably caught."""
    city, lat, lon = entity["home_city"]
    all_resources = sum(RESOURCE_POOL.values(), [])
    candidates = [r for r in all_resources if r not in entity["usual_resources"]]
    new_resources = random.sample(candidates, k=min(2, len(candidates)))
    for d in range(n_days):
        ts = start_ts + timedelta(days=d)
        hour = int(np.clip(np.random.normal(entity["hour_mean"], entity["hour_std"]), 0, 23))
        ts = ts.replace(hour=hour, minute=random.randint(0, 59))
        resource = new_resources[0] if d < n_days // 2 else random.choice(new_resources)
        rows.append({
            "timestamp": ts, "entity_id": entity["entity_id"], "entity_type": entity["entity_type"],
            "device_id": entity["device_id"], "device_fingerprint": entity["device_fingerprint"],
            "source_ip": fake.ipv4_public(), "geo_location": city, "lat": lat, "lon": lon,
            "resource_accessed": resource, "auth_method": entity["auth_method"], "session_duration": 10.0,
            "command_sequence": "login->read_file->logout",
            "login_success": True, "pattern": "insider_drift", "signal_type": "Edge case",
        })


def generate(out_path="data/access_logs.csv"):
    entities = make_entities()
    rows = []
    start = datetime.now() - timedelta(days=DAYS)

    for entity in entities:
        for d in range(DAYS):
            day_ts = start + timedelta(days=d)
            n_events = random.randint(1, 4) if entity["entity_type"] == "user" else random.randint(3, 8)
            for _ in range(n_events):
                rows.append(normal_row(entity, day_ts))

    users = [e for e in entities if e["entity_type"] == "user"]
    svc = [e for e in entities if e["entity_type"] == "service_account"]
    edge = [e for e in entities if e["entity_type"] == "edge_device"]

    # Episode counts are set so that BOTH the train window (~days 0-21) and the
    # held-out eval window (~days 22-29) end up with enough examples of every
    # class to fit and to evaluate — rare classes are bumped up more than common
    # ones since a 1-2 episode gap can otherwise fully empty one side of the split.
    for _ in range(24):
        e = random.choice(entities)
        ts = start + timedelta(days=random.randint(2, DAYS - 1))
        inject_brute_force(e, ts, rows)

    for _ in range(18):
        e = random.choice(entities)
        ts = start + timedelta(days=random.randint(2, DAYS - 1))
        inject_impossible_travel(e, ts, rows)

    for _ in range(18):
        e = random.choice(users + svc)
        ts = start + timedelta(days=random.randint(2, DAYS - 1))
        inject_lateral_movement(e, ts, rows)

    for _ in range(18):
        e = random.choice(entities)
        ts = start + timedelta(days=random.randint(2, DAYS - 1))
        inject_device_spoofing(e, ts, rows)

    for _ in range(8):
        ts = start + timedelta(days=random.randint(3, DAYS - 1))
        inject_credential_stuffing(entities, ts, rows)

    for _ in range(14):
        e = random.choice(users)
        ts = start + timedelta(days=random.randint(2, DAYS - 10))
        inject_low_and_slow_exfiltration(e, ts, rows)

    for _ in range(9):
        e = random.choice(users)
        ts = start + timedelta(days=random.randint(0, DAYS - 15))
        inject_insider_drift(e, ts, rows)

    df = pd.DataFrame(rows)
    df["label"] = df["pattern"]  # schema calls this field 'label'; hidden at inference in practice
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows across {len(entities)} entities "
          f"({len(users)} users, {len(svc)} service_accounts, {len(edge)} edge_devices) over {DAYS} days")
    print(df["signal_type"].value_counts().to_string())
    print(f"Saved to {out_path}")
    return df


if __name__ == "__main__":
    generate()
