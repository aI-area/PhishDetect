import numpy as np
import urllib.parse
import re
import logging
import math
from collections import Counter
from tldextract import extract as extract_tld_parts
from utils import UrlUtils
from constants import (
    SUSPICIOUS_TERMS,
    RARE_TLDS,
    SUSPICIOUS_DOMAINS,
    MAJOR_LEGITIMATE_SITES,
    TRUSTWORTHY_DOMAINS
)

class FeatureExtractor:
    """Class for extracting features from URLs for phishing detection."""

    FEATURE_NAMES = [
        # Lexical and Structural
        'URL Length', 'Hostname Length', 'Domain Length', 'TLD Length', 'Subdomain Length',
        'Path Length', 'Query Length', 'Max Token Length', 'Min Token Length', 'Mean Token Length',
        'Digit Count', 'Subdomain Depth', 'Domain Hierarchy', 'TLD Structure Depth', 'Encoded Hostname Indicators',
        'Slash Count', 'Dot Count', 'Hyphen in Hostname', 'Query Param Count', 'Path Depth',
        'Port Number', 'Has Port', 'Traversal Depth',

        # Ratios
        'Domain to URL Ratio', 'Subdomain to URL Ratio', 'Hostname to URL Ratio', 'Path to URL Ratio',
        'Query to URL Ratio', 'Path to Domain Ratio', 'Query to Domain Ratio', 'Query to Path Ratio',
        'Uppercase Ratio', 'Subdomain Digit Ratio',

        # Entropy
        'Hostname Entropy', 'Path Entropy', 'Query Entropy', 'URL 2-Gram Entropy', 'URL 3-Gram Entropy',
        'Hostname 2-Gram Entropy', 'Domain Entropy', 'Path Entropy Variance', 'Rolling Entropy Range',
        'Subdomain Entropy Gradient', 'Cookie Param Entropy',

        # Character Composition
        'Digits in Hostname', 'Alphabetic in Hostname', 'Punctuation in URL', 'Non-ASCII in URL',
        'Alpha Ratio in Hostname', 'Digit Ratio in Hostname', 'Special Char Ratio in Hostname',
        'Homoglyph Score',

        # Transition Patterns
        'Letter-Digit-Letter Count', 'Digit-Letter-Digit Count', 'Alpha-Digit Transitions',
        'Digit-Alpha Transitions', 'Consecutive Chars', 'Consecutive Digits', 'Consecutive Letters',
        'Special Char Cluster', 'Pattern Density',

        # Behavioral and Heuristic
        'Suspicious Keyword', 'URL Shortener', 'Digits in Subdomain', 'Redirect in Path',
        'Long Numeric Domain', 'TLD Risk Score', 'Suspicious Hosting', 'Numeric Domain Pattern',
        'Hex-Encoded IP', 'TLD-Domain Mismatch', 'Port Mismatch',

        # Protocol Features
        'IP as Hostname', 'Non-HTTP Scheme', 'WWW Prefix', 'TLD is .com', 'Punycode Domain',

        # Diversity
        'Char Diversity in URL', 'Char Diversity in Hostname',

        # Obfuscation
        'Obfuscation Chars', 'Encoding Keywords', 'Redirect Keywords', 'Redirect Chain Count',
        'Consecutive Special Chars', 'Encoding Bypass Ratio', 'Redirect TLD Disparity'
    ]

    def __init__(self):
        self.utils = UrlUtils()
        self.logger = logging.getLogger(__name__)

    def calculate_shannon_entropy(self, text):
        if not text:
            return 0
        char_frequencies = Counter(text)
        total_chars = len(text)
        probabilities = [freq / total_chars for freq in char_frequencies.values()]
        return -sum(p * math.log2(p) for p in probabilities if p > 0)

    def compute_ngram_entropy(self, text, n=2):
        if len(text) < n:
            return 0
        counts = {}
        for i in range(len(text) - n + 1):
            gram = text[i:i+n]
            counts[gram] = counts.get(gram, 0) + 1
        total = sum(counts.values())
        if total == 0:
            return 0
        return -sum(
            p * math.log2(p)
            for p in (count/total for count in counts.values())
            if p > 0
        )

    def check_ip_presence(self, hostname):
        parts = hostname.split('.')
        if len(parts) == 4 and all(part.isdigit() for part in parts):
            return all(0 <= int(p) <= 255 for p in parts)
        return False

    def calculate_tld_risk_score(self, domain, tld):
        full_domain = ""
        if domain and tld:
            full_domain = f"{domain}.{tld}".lower()
        elif domain:
            full_domain = domain.lower()
        elif tld:
            full_domain = tld.lower()

        if any(legit_site == full_domain for legit_site in MAJOR_LEGITIMATE_SITES) or \
           any(trustworthy_kw in full_domain for trustworthy_kw in TRUSTWORTHY_DOMAINS):
            return 0.0

        tld_risk_val = 0.9 if tld and ('.' + tld.lower()) in RARE_TLDS else 0.0

        if any(susp_domain in full_domain for susp_domain in SUSPICIOUS_DOMAINS):
            return 1.0

        return tld_risk_val

    def detect_url_shortener(self, url):
        parsed_local = urllib.parse.urlparse(url)
        netloc_lower = parsed_local.netloc.lower()
        shorteners = {"bit.ly", "goo.gl", "tinyurl.com", "t.co", "is.gd", "buff.ly", "adf.ly", "bc.vc"}
        if netloc_lower in shorteners:
            return 1
        patterns = [
            lambda d: len(d) <= 5 and '.' not in d,
            lambda d: bool(re.match(r'^[a-z][0-9][a-z]{1,2}\.[a-z]{2,3}$', d)),
            lambda d: (
                bool(re.match(r'^[a-z]{1,3}\.[a-z]{2,3}$', d)) and
                d.split('.')[0] not in ['com', 'org', 'net', 'edu', 'gov']
            )
        ]
        return 1 if any(p(netloc_lower) for p in patterns) else 0

    def detect_homoglyph_clusters(self, hostname):
        homoglyphs = {'а':'a', 'е':'e', 'і':'i', 'о':'o', 'с':'c'}  # Cyrillic/Latin
        return sum(1 for c in hostname if c in homoglyphs)

    def calculate_encoding_bypass(self, url):
        encoded_seq = re.findall(r'%[0-9a-fA-F]{2}', url)
        common_encodings = {'%20','%21','%2F','%3A','%3F','%3D'}
        unusual = [seq for seq in encoded_seq if seq.upper() not in common_encodings]
        return len(unusual)/len(url) if url else 0

    def path_segment_entropy_variance(self, path):
        segments = [s for s in path.split('/') if s]
        if len(segments) < 2:
            return 0
        entropies = [self.calculate_shannon_entropy(s) for s in segments]
        return np.var(entropies)

    def detect_hex_ip(self, hostname):
        hex_pattern = r'(0x[a-f0-9]{8})|(\d+\.\d+\.\d+\.\d+)'
        return 1 if re.search(hex_pattern, hostname) else 0

    def special_char_clusters(self, url):
        matches = re.findall(r'[^\w\./-]{2,}', url)
        return max(len(match) for match in matches) if matches else 0

    def rolling_entropy(self, url, window=5):
        if len(url) < window:
            return 0
        entropies = [self.calculate_shannon_entropy(url[i:i+window]) for i in range(len(url)-window)]
        return max(entropies) - min(entropies) if entropies else 0

    def tld_domain_mismatch(self, domain, tld):
        common_pairs = {'com': (3,12), 'net': (3,10), 'org': (3,8), 'io': (2,6)}
        expected = common_pairs.get(tld, (0,0))
        return 0 if expected[0] <= len(domain) <= expected[1] else 1

    def subdomain_entropy_gradient(self, subdomain):
        parts = subdomain.split('.')
        if len(parts) < 2:
            return 0
        return (
            self.calculate_shannon_entropy(parts[0]) -
            self.calculate_shannon_entropy(parts[-1])
        )

    def port_service_mismatch(self, port):
        common_ports = {80, 443, 8080, 8000, 3000}
        return 1 if port and port not in common_ports else 0

    def path_traversal_depth(self, path):
        return path.count('../') + path.count('..\\')

    def mixed_case_obfuscation(self, url):
        return sum(1 for c in url if c.isupper()) / len(url) if url else 0

    def subdomain_token_anomaly(self, subdomain):
        tokens = re.findall(r'[a-z]+|\d+', subdomain)
        return sum(1 for t in tokens if t.isdigit()) / len(tokens) if tokens else 0

    def sliding_pattern_density(self, url, window=5):
        if len(url) < window:
            return 0
        patterns = [len(set(url[i:i+window])) for i in range(len(url)-window)]
        return np.mean(patterns) if patterns else 0

    def cookie_param_entropy(self, query_str):
        if not query_str:
            return 0.0
        try:
            params = urllib.parse.parse_qs(query_str)
        except ValueError:
            return 0.0
        cookie_like_param_names = [
            "sessionid", "sessid", "sid", "token", "auth", "uid",
            "user_id", "visitorid", "clientid", "cookie", "jsessionid"
        ]
        relevant_values = []
        for key, values in params.items():
            if key.lower() in cookie_like_param_names:
                for val in values:
                    if val:
                        relevant_values.append(val)
        if not relevant_values:
            return 0.0
        concatenated_values = "".join(relevant_values)
        return self.calculate_shannon_entropy(concatenated_values)

    def redirect_tld_disparity(self, query_str, current_main_tld):
        if not query_str or not current_main_tld:
            return 0
        try:
            params = urllib.parse.parse_qs(query_str)
        except ValueError:
            return 0
        redirect_param_keys = [
            "redirect", "redirect_url", "redirecturi", "return", "return_to",
            "returnurl", "url", "goto", "target", "next", "continue", "dest", "destination"
        ]
        redirect_url_str = None
        for key, values in params.items():
            if key.lower() in redirect_param_keys:
                if values and isinstance(values[0], str):
                    if values[0].startswith('http://') or values[0].startswith('https://') or '//' in values[0]:
                        redirect_url_str = values[0]
                        break
        if not redirect_url_str:
            return 0
        try:
            if not (redirect_url_str.startswith('http://') or redirect_url_str.startswith('https://')):
                if redirect_url_str.startswith('//'):
                    redirect_url_str = 'http:' + redirect_url_str
            redirect_tld_parts = extract_tld_parts(redirect_url_str)
            redirect_tld = redirect_tld_parts.suffix.lower() if redirect_tld_parts.suffix else None
            if redirect_tld and redirect_tld != current_main_tld.lower():
                return 1
        except Exception:
            return 0
        return 0

    def generate_features(self, url, ngram_vectorizer=None, selected_ngram_indices=None):
        """Generate comprehensive features for a URL."""
        try:
            parsed_url = urllib.parse.urlparse(url)
            tld_res = extract_tld_parts(url)

            url_lower = url.lower()
            hostname = parsed_url.netloc.lower()
            path = parsed_url.path
            query = parsed_url.query
            scheme = parsed_url.scheme.lower() if parsed_url.scheme else ""
            port = parsed_url.port
            subdomain = tld_res.subdomain.lower()
            domain = tld_res.domain.lower()
            tld = tld_res.suffix.lower()

            feature_list = []

            # Lexical and Structural
            url_len = len(url)
            hostname_len = len(hostname)
            path_len = len(path)
            query_len = len(query)

            feature_list.append(url_len)           # URL Length
            feature_list.append(hostname_len)      # Hostname Length
            feature_list.append(len(domain))       # Domain Length
            feature_list.append(len(tld))          # TLD Length
            feature_list.append(len(subdomain))    # Subdomain Length
            feature_list.append(path_len)          # Path Length
            feature_list.append(query_len)         # Query Length

            tokens = url.split('/')
            feature_list.append(
                max(len(token) for token in tokens) if tokens and any(tokens) else 0
            )  # Max Token Length
            feature_list.append(
                min(len(token) for token in tokens if token) if tokens and any(token for token in tokens if token) else 0
            )  # Min Token Length
            feature_list.append(
                np.mean([len(token) for token in tokens if token]) if tokens and any(token for token in tokens if token) else 0
            )  # Mean Token Length
            feature_list.append(sum(c.isdigit() for c in url))  # Digit Count

            feature_list.append(
                subdomain.count('.') + 1 if subdomain else 0
            )  # Subdomain Depth
            feature_list.append(
                domain.count('.') + 1 if domain else 0
            )  # Domain Hierarchy
            feature_list.append(
                tld.count('.') + 1 if tld else 0
            )  # TLD Structure Depth

            non_alnum_hostname = sum(not c.isalnum() for c in hostname)
            feature_list.append(non_alnum_hostname)  # Encoded Hostname Indicators

            feature_list.append(url.count('/'))       # Slash Count
            feature_list.append(url.count('.'))       # Dot Count
            feature_list.append(hostname.count('-'))  # Hyphen in Hostname
            feature_list.append(
                len(query.split('&')) if query else 0
            )  # Query Param Count
            feature_list.append(path.count('/'))     # Path Depth
            feature_list.append(
                port if port is not None else -1
            )  # Port Number
            feature_list.append(
                1 if port is not None else 0
            )  # Has Port
            feature_list.append(self.path_traversal_depth(path))  # Traversal Depth

            # Ratios
            feature_list.append(
                len(domain) / url_len if url_len > 0 else 0
            )  # Domain to URL Ratio
            feature_list.append(
                len(subdomain) / url_len if url_len > 0 else 0
            )  # Subdomain to URL Ratio
            feature_list.append(
                hostname_len / url_len if url_len > 0 else 0
            )  # Hostname to URL Ratio
            feature_list.append(
                path_len / url_len if url_len > 0 else 0
            )  # Path to URL Ratio
            feature_list.append(
                query_len / url_len if url_len > 0 else 0
            )  # Query to URL Ratio
            feature_list.append(
                path_len / len(domain) if len(domain) > 0 else 0
            )  # Path to Domain Ratio
            feature_list.append(
                query_len / len(domain) if len(domain) > 0 else 0
            )  # Query to Domain Ratio
            feature_list.append(
                query_len / path_len if path_len > 0 else 0
            )  # Query to Path Ratio
            feature_list.append(self.mixed_case_obfuscation(url))      # Uppercase Ratio
            feature_list.append(self.subdomain_token_anomaly(subdomain))  # Subdomain Digit Ratio

            # Entropy
            feature_list.append(self.calculate_shannon_entropy(hostname))      # Hostname Entropy
            feature_list.append(self.calculate_shannon_entropy(path))          # Path Entropy
            feature_list.append(self.calculate_shannon_entropy(query))         # Query Entropy
            feature_list.append(self.compute_ngram_entropy(url, n=2))          # URL 2‑Gram Entropy
            feature_list.append(self.compute_ngram_entropy(url, n=3))          # URL 3‑Gram Entropy
            feature_list.append(self.compute_ngram_entropy(hostname, n=2))     # Hostname 2‑Gram Entropy
            feature_list.append(self.calculate_shannon_entropy(domain))        # Domain Entropy
            feature_list.append(self.path_segment_entropy_variance(path))      # Path Entropy Variance
            feature_list.append(self.rolling_entropy(url))                      # Rolling Entropy Range
            feature_list.append(self.subdomain_entropy_gradient(subdomain))    # Subdomain Entropy Gradient
            feature_list.append(self.cookie_param_entropy(query))              # Cookie Param Entropy

            # Character Composition
            digits_hostname = sum(c.isdigit() for c in hostname)
            letters_hostname = sum(c.isalpha() for c in hostname)
            feature_list.append(digits_hostname)            # Digits in Hostname
            feature_list.append(letters_hostname)           # Alphabetic in Hostname
            feature_list.append(
                sum(1 for c in url if c in "!@#$%^&*()")
            )  # Punctuation in URL
            feature_list.append(
                sum(ord(c) > 127 for c in url)
            )  # Non‑ASCII in URL
            feature_list.append(
                letters_hostname / hostname_len if hostname_len > 0 else 0
            )  # Alpha Ratio in Hostname
            feature_list.append(
                digits_hostname / hostname_len if hostname_len > 0 else 0
            )  # Digit Ratio in Hostname
            feature_list.append(
                non_alnum_hostname / hostname_len if hostname_len > 0 else 0
            )  # Special Char Ratio in Hostname
            feature_list.append(self.detect_homoglyph_clusters(hostname))  # Homoglyph Score

            # Transition Patterns
            url_len_nonzero = url_len if url_len > 0 else 1
            letter_digit_letter = sum(
                1
                for i in range(1, url_len_nonzero - 1)
                if url[i].isdigit() and url[i - 1].isalpha() and url[i + 1].isalpha()
            )
            digit_letter_digit = sum(
                1
                for i in range(1, url_len_nonzero - 1)
                if url[i].isalpha() and url[i - 1].isdigit() and url[i + 1].isdigit()
            )
            feature_list.append(letter_digit_letter)  # Letter‑Digit‑Letter Count
            feature_list.append(digit_letter_digit)   # Digit‑Letter‑Digit Count
            feature_list.append(
                sum(1 for i in range(url_len_nonzero - 1) if url[i].isalpha() and url[i + 1].isdigit()) / url_len_nonzero
            )  # Alpha‑Digit Transitions
            feature_list.append(
                sum(1 for i in range(url_len_nonzero - 1) if url[i].isdigit() and url[i + 1].isalpha()) / url_len_nonzero
            )  # Digit‑Alpha Transitions
            feature_list.append(
                sum(1 for i in range(url_len_nonzero - 1) if url[i] == url[i + 1])
            )  # Consecutive Chars
            feature_list.append(
                sum(1 for i in range(url_len_nonzero - 1) if url[i].isdigit() and url[i + 1].isdigit())
            )  # Consecutive Digits
            feature_list.append(
                sum(1 for i in range(url_len_nonzero - 1) if url[i].isalpha() and url[i + 1].isalpha())
            )  # Consecutive Letters
            feature_list.append(self.special_char_clusters(url))      # Special Char Cluster
            feature_list.append(self.sliding_pattern_density(url))    # Pattern Density

            # Behavioral and Heuristic
            feature_list.append(
                1 if any(keyword.lower() in url_lower for keyword in SUSPICIOUS_TERMS) else 0
            )  # Suspicious Keyword
            feature_list.append(self.detect_url_shortener(url))  # URL Shortener
            feature_list.append(
                1 if any(c.isdigit() for c in subdomain) else 0
            )  # Digits in Subdomain
            feature_list.append(
                1 if ("//" in path or "redirect" in path.lower()) else 0
            )  # Redirect in Path
            feature_list.append(
                1 if domain and (len(domain) > 8 and any(c.isdigit() for c in domain)) else 0
            )  # Long Numeric Domain
            feature_list.append(self.calculate_tld_risk_score(domain, tld))     # TLD Risk Score
            feature_list.append(
                1 if any(susp_domain in hostname for susp_domain in SUSPICIOUS_DOMAINS) else 0
            )  # Suspicious Hosting

            numeric_pattern = False
            if domain:
                digit_sum_domain = sum(c.isdigit() for c in domain)
                if ( (digit_sum_domain / len(domain) > 0.3 if len(domain) > 0 else False ) or digit_sum_domain > 3 ):
                    numeric_pattern = True
            feature_list.append(1 if numeric_pattern else 0)  # Numeric Domain Pattern

            feature_list.append(self.detect_hex_ip(hostname))               # Hex‑Encoded IP
            feature_list.append(self.tld_domain_mismatch(domain, tld))     # TLD‑Domain Mismatch
            feature_list.append(self.port_service_mismatch(port))          # Port Mismatch

            # Protocol Features
            feature_list.append(
                1 if self.check_ip_presence(hostname) else 0
            )  # IP as Hostname
            feature_list.append(
                0 if scheme == "http" else 1
            )  # Non‑HTTP Scheme
            feature_list.append(
                1 if hostname.startswith("www.") else 0
            )  # WWW Prefix
            feature_list.append(
                1 if tld == "com" else 0
            )  # TLD is .com
            feature_list.append(
                1 if domain.startswith("xn--") else 0
            )  # Punycode Domain

            # Diversity
            feature_list.append(
                len(set(url)) / url_len if url_len > 0 else 0
            )  # Char Diversity in URL
            feature_list.append(
                len(set(hostname)) / hostname_len if hostname_len > 0 else 0
            )  # Char Diversity in Hostname

            # Obfuscation
            feature_list.append(
                1 if "%" in url or any(ord(c) > 127 for c in url) else 0
            )  # Obfuscation Chars
            feature_list.append(
                1 if re.search(r"(base64|hex)[^a-zA-Z0-9]", url, re.IGNORECASE) else 0
            )  # Encoding Keywords
            feature_list.append(
                1 if (query and any(redir in query.lower() for redir in ["redirect", "url=", "goto=", "forward="])) else 0
            )  # Redirect Keywords
            feature_list.append(
                url.count("//") - 1 if (url.count("//") > 0 and "://" in url) else url.count("//")
            )  # Redirect Chain Count
            feature_list.append(
                sum(1 for i in range(url_len - 1) if not url[i].isalnum() and url[i] == url[i + 1])
            )  # Consecutive Special Chars
            feature_list.append(self.calculate_encoding_bypass(url))  # Encoding Bypass Ratio
            feature_list.append(self.redirect_tld_disparity(query, tld))  # Redirect TLD Disparity

            # N-gram Features (only the selected indices)
            if (ngram_vectorizer is not None) and (selected_ngram_indices is not None):
                ngram_array = ngram_vectorizer.transform([url]).toarray().flatten()
                # Safely pick only those indices that exist
                valid_indices = [
                    idx for idx in selected_ngram_indices
                    if idx < len(ngram_array)
                ]
                final_vals = np.zeros(len(selected_ngram_indices), dtype=np.float32)
                for i, idx in enumerate(selected_ngram_indices):
                    if idx < len(ngram_array):
                        final_vals[i] = ngram_array[idx]
                feature_list.extend(final_vals)

            return feature_list

        except Exception as e:
            self.logger.error(f"Error generating features for URL '{url}': {e}")
            return []


