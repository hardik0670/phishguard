import re
import math
from urllib.parse import urlparse

KNOWN_BRANDS = {
    "google": {"google.com"},
    "gmail": {"gmail.com", "googlemail.com"},
    "apple": {"apple.com"},
    "amazon": {"amazon.com"},
    "microsoft": {"microsoft.com", "live.com", "outlook.com"},
    "paypal": {"paypal.com"},
    "ebay": {"ebay.com"},
    "facebook": {"facebook.com", "fb.com"},
    "instagram": {"instagram.com"},
    "netflix": {"netflix.com"},
    "wellsfargo": {"wellsfargo.com"},
    "chase": {"chase.com"},
    "bankofamerica": {"bankofamerica.com"},
}

COMMON_MULTI_PART_SUFFIXES = {
    "co.uk", "com.au", "com.br", "com.cn", "com.hk", "com.mx", "com.sg",
    "co.in", "co.jp", "co.nz", "co.za", "org.uk", "net.au",
}

BRAND_KEYWORDS = set(KNOWN_BRANDS)

TYPO_BRANDS = {
    "google", "gmail", "paypal", "amazon", "apple", "microsoft", "facebook",
    "instagram", "netflix", "ebay",
}

def get_entropy(s):
    if not s:
        return 0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    return -sum((f/len(s)) * math.log2(f/len(s)) for f in freq.values())

def normalize_hostname(hostname):
    host = hostname.lower().strip(".")
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    if ":" in host and not re.match(r"^\d{1,3}(\.\d{1,3}){3}(:\d+)?$", host):
        host = host.split(":", 1)[0]
    return host[4:] if host.startswith("www.") else host

def get_registrable_domain(hostname):
    host = normalize_hostname(hostname)
    parts = [p for p in host.split(".") if p]
    if len(parts) <= 2:
        return host

    suffix = ".".join(parts[-2:])
    if suffix in COMMON_MULTI_PART_SUFFIXES and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])

def get_domain_label(hostname):
    registered = get_registrable_domain(hostname)
    return registered.split(".", 1)[0]

def edit_distance(a, b):
    prev_prev = None
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            value = min(
                cur[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + cost
            )
            if (
                prev_prev is not None and i > 1 and j > 1
                and ca == b[j - 2] and a[i - 2] == cb
            ):
                value = min(value, prev_prev[j - 2] + 1)
            cur.append(value)
        prev_prev = prev
        prev = cur
    return prev[-1]

def is_official_brand_domain(hostname):
    registered = get_registrable_domain(hostname)
    return any(registered in domains for domains in KNOWN_BRANDS.values())

def detect_brand_risk(url):
    parsed = urlparse(url if url.startswith("http") else "http://" + url)
    host = normalize_hostname(parsed.hostname or parsed.netloc)
    registered = get_registrable_domain(host)
    label = get_domain_label(host)
    full_url = url.lower()

    official_brand = None
    for brand, domains in KNOWN_BRANDS.items():
        if registered in domains:
            official_brand = brand
            break

    suspicious_brand = None
    suspicious_reason = None

    for brand, domains in KNOWN_BRANDS.items():
        if official_brand == brand:
            continue

        brand_mentions = brand in full_url or any(domain in full_url for domain in domains)
        label_tokens = [token for token in re.split(r"[^a-z0-9]+", label) if token]
        typo_match = any(
            token != brand and abs(len(token) - len(brand)) <= 2 and edit_distance(token, brand) <= 1
            for token in label_tokens + [label]
            if len(token) >= max(4, len(brand) - 1)
        )

        if brand_mentions:
            suspicious_brand = brand
            suspicious_reason = f"{brand} mentioned outside official domain"
            break
        if brand in TYPO_BRANDS and typo_match:
            suspicious_brand = brand
            suspicious_reason = f"possible {brand} lookalike domain"
            break

    return {
        "registered_domain": registered,
        "domain_label": label,
        "official_brand": official_brand,
        "suspicious_brand": suspicious_brand,
        "suspicious_reason": suspicious_reason,
        "has_brand_mimicry": 1 if suspicious_brand else 0,
    }

def has_phishing_keyword(url):
    risk = detect_brand_risk(url)
    if risk["official_brand"]:
        non_brand_words = r"login|secure|verify|update|bank|account|confirm|signin|wallet|crypto|free|gift|prize|win|claim|support|help|security|alert"
        return 1 if re.search(non_brand_words, url, re.IGNORECASE) else 0
    return 1 if re.search(
        r"login|secure|verify|update|bank|paypal|gmail|google|account|confirm|signin|ebay|wallet|crypto|free|gift|prize|win|claim|support|help|security|alert",
        url, re.IGNORECASE) else 0

def extract_features(url):
    try:
        parsed = urlparse(url if url.startswith("http") else "http://" + url)
        domain = normalize_hostname(parsed.hostname or parsed.netloc)
        path = parsed.path.lower()
        full_url = url.lower()

        # Suspicious TLDs
        suspicious_tlds = {".xyz", ".top", ".work", ".bid", ".info", ".icu", ".buzz", ".tk", ".ml", ".ga", ".cf", ".gq"}
        has_suspicious_tld = 1 if any(domain.endswith(tld) for tld in suspicious_tlds) else 0

        brand_risk = detect_brand_risk(full_url)
        has_brand_mimicry = brand_risk["has_brand_mimicry"]

        # Shortened URLs
        shorteners = ["bit.ly", "t.co", "tinyurl.com", "is.gd", "goo.gl", "buff.ly", "adf.ly"]
        is_shortened = 1 if any(s in domain for s in shorteners) else 0

        # Character counts
        digits = sum(c.isdigit() for c in full_url)
        letters = sum(c.isalpha() for c in full_url)
        vowels = sum(c in "aeiou" for c in full_url)
        
        return {
            "url_length":          len(url),
            "has_https":           1 if parsed.scheme == "https" else 0,
            "has_ip":              1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain) else 0,
            "dot_count":           url.count("."),
            "hyphen_count":        url.count("-"),
            "at_count":            url.count("@"),
            "slash_count":         url.count("/"),
            "question_count":      url.count("?"),
            "equal_count":         url.count("="),
            "underscore_count":    url.count("_"),
            "digit_count":         digits,
            "special_char_count":  len(re.findall(r"[^a-zA-Z0-9]", url)),
            "subdomain_count":     len(domain.split(".")) - 2 if domain else 0,
            "path_length":         len(path),
            "entropy":             round(get_entropy(url), 4),
            "has_phishing_words":  has_phishing_keyword(url),
            "url_depth":           len([p for p in path.split("/") if p]),
            "has_port":            1 if parsed.port else 0,
            "domain_length":       len(domain),
            "has_double_slash":    1 if "//" in path else 0,
            "tld_in_path":         1 if re.search(r"\.(com|net|org|php|html|asp|aspx|js)", path) else 0,
            "has_hex_encoding":    1 if re.search(r"%[0-9a-fA-F]{2}", url) else 0,
            "digit_ratio":         round(digits / max(len(url), 1), 4),
            "letter_ratio":        round(letters / max(len(url), 1), 4),
            "vowel_ratio":         round(vowels / max(letters, 1), 4),
            "has_suspicious_tld":  has_suspicious_tld,
            "has_brand_mimicry":   has_brand_mimicry,
            "is_shortened":        is_shortened,
            "has_redirection":     1 if "http" in path or "http" in parsed.query else 0,
        }
    except:
        return None
