"""Generate a small synthetic dataset in the exact challenge format.

This is ONLY for developing / testing the pipeline before the real data is
available. It mimics the noise patterns described in the problem statement:
abbreviations, legal-suffix changes, '&' vs 'and', typos, word-order swaps,
missing address components, landmark references, and hard negatives (distinct
businesses with near-identical names in different places). Train covers US
and India; test additionally contains France.

Usage: python make_fake_data.py --out ../fake_dataset --n-train 3000 --n-test 1500
"""
import argparse
import os
import random

WORDS = ["Sunrise", "Global", "Apex", "Blue", "River", "Summit", "Golden", "Star", "Green",
         "Metro", "Prime", "Royal", "Silver", "Eagle", "Pioneer", "Liberty", "Crystal", "Orion",
         "Lotus", "Harbor", "Maple", "Cedar", "Falcon", "Atlas", "Nova", "Vertex", "Delta",
         "Crescent", "Evergreen", "Horizon", "Unity", "Zenith", "Coastal", "Northern", "Pacific"]
IN_WORDS = ["Sharma", "Gupta", "Shree", "Sai", "Ganesh", "Laxmi", "Balaji", "Krishna", "Patel",
            "Mehta", "Agarwal", "Bharat", "Om", "Durga", "Annapurna", "Raj", "Vinayak", "Siddhi"]
FR_WORDS = ["Boulangerie", "Maison", "Atelier", "Dupont", "Lefèvre", "Moreau", "Château",
            "Étoile", "Rivière", "Soleil", "Garçon", "Brasserie", "Côte", "Martin", "Bernard"]
TRADES = ["Traders", "Technologies", "Solutions", "Services", "Foods", "Motors", "Textiles",
          "Pharmaceuticals", "Consulting", "Logistics", "Electronics", "Builders", "Associates",
          "Industries", "Systems", "Restaurant", "Hardware", "Jewellers", "Clinic"]
US_SUFFIX = [("Inc", "Incorporated", "Inc."), ("LLC", "L.L.C.", "LLC"), ("Corp", "Corporation", "Corp."),
             ("Co", "Company", "Co.")]
IN_SUFFIX = [("Pvt Ltd", "Private Limited", "Pvt. Ltd."), ("Ltd", "Limited", "Ltd.")]
FR_SUFFIX = [("SARL", "S.A.R.L.", "Sarl"), ("SAS", "S.A.S.", "SAS"), ("SA", "S.A.", "SA")]
US_CITIES = [("Austin", "TX"), ("Denver", "CO"), ("Seattle", "WA"), ("Boston", "MA"),
             ("Phoenix", "AZ"), ("Chicago", "IL"), ("Miami", "FL"), ("Portland", "OR")]
US_STREETS = ["Main", "Oak", "Pine", "Maple", "Cedar", "Elm", "Washington", "Lake", "Hill", "Park"]
US_TYPES = [("Street", "St"), ("Road", "Rd"), ("Avenue", "Ave"), ("Boulevard", "Blvd"), ("Drive", "Dr")]
IN_CITIES = [("Bengaluru", "Karnataka", "Bangalore"), ("Mumbai", "Maharashtra", "Bombay"),
             ("Chennai", "Tamil Nadu", "Madras"), ("Pune", "Maharashtra", "Poona"),
             ("Kolkata", "West Bengal", "Calcutta"), ("Hyderabad", "Telangana", "Hyderabad")]
IN_AREAS = ["MG Road", "Koramangala", "Andheri East", "T Nagar", "Banjara Hills", "Salt Lake",
            "Indiranagar", "Shivaji Nagar", "Gandhi Nagar", "Sector 17"]
IN_LANDMARKS = ["Near SBI ATM", "Opp. Bus Stand", "Behind City Mall", "Near Railway Station",
                "Opposite Apollo Hospital"]
FR_CITIES = [("Paris", "75"), ("Lyon", "69"), ("Marseille", "13"), ("Toulouse", "31"), ("Lille", "59")]
FR_STREETS = ["de la République", "Victor Hugo", "Jean Jaurès", "de la Paix", "Pasteur", "Gambetta"]
FR_TYPES = [("Rue", "R."), ("Avenue", "Av."), ("Boulevard", "Bd"), ("Place", "Pl.")]


def typo(s, rng, p=0.5):
    """Randomly insert / delete / swap / substitute one character."""
    if len(s) < 4 or rng.random() > p:
        return s
    i = rng.randrange(1, len(s) - 1)
    op = rng.choice("idsr")
    c = rng.choice("abcdefghijklmnopqrstuvwxyz")
    if op == "i":
        return s[:i] + c + s[i:]
    if op == "d":
        return s[:i] + s[i + 1:]
    if op == "s":
        return s[:i - 1] + s[i] + s[i - 1] + s[i + 1:]
    return s[:i] + c + s[i + 1:]


def make_entity(country, rng):
    """A clean base business: dict with name parts and address parts."""
    if country == "US":
        base = rng.sample(WORDS, rng.choice([1, 2]))
        suf = rng.choice(US_SUFFIX)
        city, st = rng.choice(US_CITIES)
        addr = dict(num=str(rng.randint(10, 9999)), street=rng.choice(US_STREETS),
                    stype=rng.choice(US_TYPES), city=city, state=st, pc=str(rng.randint(10000, 99999)))
    elif country == "India":
        base = rng.sample(IN_WORDS + WORDS, rng.choice([1, 2]))
        suf = rng.choice(IN_SUFFIX)
        city = rng.choice(IN_CITIES)
        addr = dict(num=f"{rng.randint(1, 300)}/{rng.randint(1, 20)}", area=rng.choice(IN_AREAS),
                    city=city, pc=str(rng.randint(110000, 859999)), lm=rng.choice(IN_LANDMARKS))
    else:
        base = rng.sample(FR_WORDS, rng.choice([1, 2]))
        suf = rng.choice(FR_SUFFIX)
        city, dep = rng.choice(FR_CITIES)
        addr = dict(num=str(rng.randint(1, 180)), street=rng.choice(FR_STREETS),
                    stype=rng.choice(FR_TYPES), city=city, pc=dep + f"{rng.randint(0, 999):03d}")
    trade = rng.choice(TRADES) if country != "France" else rng.choice(["", "et Fils", "Services", "Conseil"])
    return dict(country=country, base=base, trade=trade, suf=suf, addr=addr)


def render(e, rng, noisy):
    """Render an entity as (name, address) with optional noise."""
    c = e["country"]
    base = list(e["base"])
    if noisy and len(base) > 1 and rng.random() < 0.15:
        base.reverse()
    if noisy and rng.random() < 0.15 and len(base) > 1:
        base = [base[0]] + ["&"] if rng.random() < 0.5 else base
    parts = base + ([e["trade"]] if e["trade"] else [])
    if noisy:
        parts = [typo(p, rng, 0.2) for p in parts]
        if rng.random() < 0.2:
            parts = [p.upper() for p in parts]
    suf = e["suf"][rng.randrange(3)] if noisy else e["suf"][0]
    if noisy and rng.random() < 0.25:
        suf = ""
    name = " ".join(parts + ([suf] if suf else []))
    a = e["addr"]
    if c == "US":
        stype = a["stype"][rng.randrange(2)] if noisy else a["stype"][0]
        comps = [f"{a['num']} {a['street']} {stype}", a["city"], f"{a['state']} {a['pc']}"]
        if noisy and rng.random() < 0.2:
            comps[2] = a["state"]
        if noisy and rng.random() < 0.1:
            comps = comps[:1]
    elif c == "India":
        city = a["city"][2] if noisy and rng.random() < 0.3 else a["city"][0]
        pc = a["pc"] if not noisy or rng.random() > 0.3 else ""
        pc = pc[:3] + " " + pc[3:] if pc and noisy and rng.random() < 0.3 else pc
        comps = [a["num"], a["area"], city]
        if noisy and rng.random() < 0.4:
            comps.insert(2, a["lm"])
        if not noisy or rng.random() > 0.3:
            comps.append(a["city"][1])
        if pc:
            comps.append(pc)
        if noisy and rng.random() < 0.15:
            comps[0], comps[1] = comps[1], comps[0]
    else:
        stype = a["stype"][rng.randrange(2)] if noisy else a["stype"][0]
        comps = [f"{a['num']} {stype} {a['street']}", f"{a['pc']} {a['city']}"]
        if noisy and rng.random() < 0.3:
            comps[1] = a["city"]
    addr = ", ".join(comps)
    if noisy:
        addr = typo(addr, rng, 0.2)
    return name, addr


def hard_negative(e, rng):
    """Different business, same name, different address (tests precision)."""
    f = make_entity(e["country"], rng)
    f["base"], f["trade"], f["suf"] = e["base"], e["trade"], e["suf"]
    return f


def gen_split(n, countries, rng, prefix_off=0):
    """Generate S1/S2/S3 rows and ground truth for one split."""
    s1, s2, s3, gt = [], [], [], []
    ents = []
    for _ in range(n):
        c = rng.choice(countries)
        e = make_entity(c, rng)
        ents.append(e)
        if rng.random() < 0.08:  # a near-duplicate *distinct* business in S1
            ents.append(hard_negative(e, rng))
    rng.shuffle(ents)
    for i, e in enumerate(ents):
        s1.append((e, render(e, rng, noisy=False)))
    # matches: 0..3 across S2/S3; ~30% singletons
    other2, other3 = [], []
    links = []
    for i, e in enumerate(ents):
        r = rng.random()
        k = 0 if r < 0.3 else (1 if r < 0.65 else (2 if r < 0.9 else 3))
        for _ in range(k):
            links.append((i, rng.choice([2, 3]), e))
    # distractor S2/S3 records that match nothing in S1
    for _ in range(n // 4):
        c = rng.choice(countries)
        e = make_entity(c, rng)
        links.append((-1, rng.choice([2, 3]), e))
    for _ in range(n // 10):  # hard-negative distractors: same name, other address
        e = rng.choice(ents)
        links.append((-1, rng.choice([2, 3]), hard_negative(e, rng)))
    rng.shuffle(links)
    matches = {i: [] for i in range(len(ents))}
    c2 = c3 = 0
    for i, src, e in links:
        name, addr = render(e, rng, noisy=True)
        if src == 2:
            c2 += 1
            eid = f"S2-{c2:05d}"
            s2.append((eid, name, addr, e["country"]))
        else:
            c3 += 1
            eid = f"S3-{c3:05d}"
            s3.append((eid, name, addr, e["country"]))
        if i >= 0:
            matches[i].append(eid)
    rows1 = [(f"S1-{i + 1:05d}", nm, ad, e["country"]) for i, (e, (nm, ad)) in enumerate(s1)]
    gt = [(rows1[i][0], ",".join(matches[i])) for i in range(len(ents))]
    return rows1, s2, s3, gt


def write(path, header, rows):
    """Write rows as a TSV with the given header."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fake_dataset")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    H = ["entity_id", "business_name", "business_address", "country"]
    for split, n, cs in [("train", a.n_train, ["US", "India"]),
                         ("test", a.n_test, ["US", "India", "France"])]:
        d = os.path.join(a.out, split)
        os.makedirs(d, exist_ok=True)
        r1, r2, r3, gt = gen_split(n, cs, rng)
        write(os.path.join(d, f"{split}_source1.tsv"), H, r1)
        write(os.path.join(d, f"{split}_source2.tsv"), H, r2)
        write(os.path.join(d, f"{split}_source3.tsv"), H, r3)
        if split == "train":
            write(os.path.join(d, "train_ground_truth.tsv"), ["source1_entity_id", "matched_entity_ids"], gt)
        else:  # keep test labels separately so we can sanity-check France behaviour
            write(os.path.join(a.out, "test_ground_truth_FAKE_ONLY.tsv"),
                  ["source1_entity_id", "matched_entity_ids"], gt)
        print(split, len(r1), len(r2), len(r3))


if __name__ == "__main__":
    main()
