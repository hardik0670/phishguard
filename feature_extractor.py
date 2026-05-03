import re
import math
from urllib.parse import urlparse

def get_entropy(s):
    if not s:
        return 0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    return -sum((f/len(s)) * math.log2(f/len(s)) for f in freq.values())

def extract_features(url):
    try:
        parsed = urlparse(url if url.startswith("http") else "http://" + url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        full_url = url.lower()

        # Suspicious TLDs
        suspicious_tlds = {".xyz", ".top", ".work", ".bid", ".info", ".icu", ".buzz", ".tk", ".ml", ".ga", ".cf", ".gq"}
        has_suspicious_tld = 1 if any(domain.endswith(tld) for tld in suspicious_tlds) else 0

        # Brand mimicking detection (checking if brand names appear but are NOT the primary domain)
        brands = ["google", "apple", "amazon", "microsoft", "paypal", "ebay", "facebook", "instagram", "netflix", "wellsfargo", "chase", "bankofamerica"]
        # A simple way to check if a brand is in the URL but the domain isn't primarily that brand
        has_brand_mimicry = 0
        for brand in brands:
            if brand in full_url:
                # If brand is in url but domain is not just 'brand.com' or 'brand.co.uk' etc.
                # This is a bit naive but effective for many phishing cases
                if brand not in domain.split(".")[0]:
                    has_brand_mimicry = 1
                    break

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
            "has_phishing_words":  1 if re.search(
                r"login|secure|verify|update|bank|paypal|account|confirm|signin|ebay|wallet|crypto|free|gift|prize|win|claim|support|help|security|alert",
                url, re.IGNORECASE) else 0,
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