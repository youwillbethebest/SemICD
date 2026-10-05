"""SID tokens: register them in the tokenizer, encode code sets, and constrain decoding."""
import re

SEP = '<SEP>'
END = '<END>'


def code_tokens(values):
    result = [f'<SID{level}_{value}>' for level, value in enumerate(values[:3], 1)]
    return result + ([f'<UID_{values[3]}>'] if len(values) == 4 else [])


def register_bundle(tokenizer, model, bundle):
    representation = bundle['manifest'].get('representation', 'semantic')
    if representation == 'raw':
        return make_codec(tokenizer, bundle)
    if representation == 'atomic':
        tokens = [f'<ATOM_{v[0]:05d}>' for v in bundle['assignments'].values()]
    else:
        tokens = [f'<SID{level}_{value}>' for level in range(1, 4) for value in range(128)]
        tokens += [f'<UID_{v}>' for v in sorted({path[3] for path in bundle['assignments'].values() if len(path) == 4})]
    tokenizer.add_special_tokens({'additional_special_tokens': tokens + [SEP, END]})
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    return make_codec(tokenizer, bundle)


class Representation:
    """Shared exact-set codec and canonical constrained decoder for SID/Atomic."""
    constrained = True
    def __init__(self, tokenizer, bundle):
        self.tokenizer, self.bundle = tokenizer, bundle
        self.sep, self.end = self.single(SEP), self.single(END)
        self.forward = {
            code: tuple(self.single(t) for t in self.tokens(v))
            for code, v in bundle['assignments'].items()
        }
        self.reverse = {v: k for k, v in self.forward.items()}
        self.canonical_codes = tuple(sorted(self.forward))
        self.code_rank = {code: rank for rank, code in enumerate(self.canonical_codes)}
        self.path_rank = {self.forward[code]: self.code_rank[code] for code in self.canonical_codes}

        # Retain a trie for inspection and build a rank-aware prefix index for decoding.
        self.trie = {}
        self.next_token_max_rank = {}
        for code in self.canonical_codes:
            sequence = self.forward[code]
            rank = self.code_rank[code]
            node = self.trie
            for position, token in enumerate(sequence):
                prefix = sequence[:position]
                by_token = self.next_token_max_rank.setdefault(prefix, {})
                by_token[token] = max(rank, by_token.get(token, -1))
                node = node.setdefault(token, {})
            node[None] = True

    def single(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1 or self.tokenizer.convert_ids_to_tokens(ids[0]) != text:
            raise ValueError(f'Missing atomic token: {text}')
        return ids[0]

    def encode(self, codes):
        result = []
        for code in sorted(set(codes)):
            if code not in self.forward:
                raise ValueError(f'Unknown code: {code}')
            if result:
                result.append(self.sep)
            result.extend(self.forward[code])
        return result + [self.end]

    def _parse_prefix(self, generated):
        values = tuple(map(int, generated))
        if self.end in values:
            if values[-1] != self.end or values.count(self.end) != 1:
                raise ValueError('END may occur only once at the end')
            return (), -1, True

        completed = []
        current = []
        trailing_separator = False
        for token in values:
            if token == self.sep:
                if not current:
                    raise ValueError('SEP requires a complete preceding SID')
                path = tuple(current)
                if path not in self.path_rank:
                    raise ValueError('SEP followed an incomplete or unknown SID')
                completed.append(self.path_rank[path])
                current = []
                trailing_separator = True
            else:
                current.append(token)
                trailing_separator = False

        if any(right <= left for left, right in zip(completed, completed[1:])):
            raise ValueError('Completed SIDs are duplicated or out of canonical order')
        last_rank = completed[-1] if completed else -1
        return tuple(current), last_rank, trailing_separator

    def decode(self, ids):
        values = tuple(map(int, ids))
        if not values or values[-1] != self.end or values.count(self.end) != 1:
            return [], False, False

        body = values[:-1]
        if not body:
            return [], True, True

        segments, current = [], []
        for token in body:
            if token == self.sep:
                if not current:
                    return [], False, False
                segments.append(tuple(current))
                current = []
            else:
                current.append(token)
        if not current:
            return [], False, False
        segments.append(tuple(current))

        if any(path not in self.reverse for path in segments):
            return [], False, False
        ranks = [self.path_rank[path] for path in segments]
        if any(right <= left for left, right in zip(ranks, ranks[1:])):
            return [], False, False
        return [self.reverse[path] for path in segments], True, True

    def allowed(self, generated):
        values = tuple(map(int, generated))
        if self.end in values:
            if values[-1] == self.end and values.count(self.end) == 1:
                return [self.end]
            raise ValueError('END may occur only once at the end')

        current, last_rank, trailing_separator = self._parse_prefix(values)

        if current in self.path_rank:
            rank = self.path_rank[current]
            if rank <= last_rank:
                raise ValueError('SID violates canonical order')
            allowed = [self.end]
            if rank + 1 < len(self.canonical_codes):
                allowed.append(self.sep)
            return allowed

        by_token = self.next_token_max_rank.get(current, {})
        allowed = [
            token for token, maximum_rank in by_token.items()
            if maximum_rank > last_rank
        ]
        if not allowed:
            raise ValueError('Generated tokens are not a prefix of a legal SID')
        if not current and not trailing_separator:
            allowed.append(self.end)
        return sorted(set(allowed))

    def tokens(self, values):
        return code_tokens(values)


class SemanticSIDCodec(Representation):
    pass


class RandomSIDCodec(SemanticSIDCodec):
    pass


class AtomicIDCodec(Representation):
    def tokens(self, values):
        return [f'<ATOM_{values[0]:05d}>']


class RawCodec(Representation):
    """Native code text and EOS; Raw generation has no SID prefix constraint."""
    constrained = False

    def __init__(self, tokenizer, bundle):
        self.tokenizer, self.bundle = tokenizer, bundle
        self.end = tokenizer.eos_token_id
        if self.end is None:
            raise ValueError('Raw requires native EOS')
        self.forward = {c: tuple(tokenizer.encode(c, add_special_tokens=False)) for c in bundle['assignments']}

    def encode(self, codes):
        codes = sorted(set(codes))
        if set(codes) - set(self.forward):
            raise ValueError('Unknown code in Raw training target')
        text = '<code>' + ','.join(codes) + '</code>'
        return self.tokenizer.encode(text, add_special_tokens=False) + [self.end]

    def decode(self, ids):
        ids = list(map(int, ids))
        complete = self.end in ids
        body = ids[:ids.index(self.end)] if complete else ids
        text = self.tokenizer.decode(body, skip_special_tokens=False).strip()
        match = re.fullmatch(r'<code>(.*?)</code>', text, flags=re.DOTALL)
        if match is None:
            return [], complete, False
        pieces = [c.strip() for c in match.group(1).strip().split(',')] if match.group(1).strip() else []
        if any(not c for c in pieces):
            return [], complete, False
        codes = [c.upper().replace('.', '') for c in pieces] if self.bundle['manifest']['code_system'] != 'synthetic' else pieces
        # Match main: valid code-tag format is independent of catalog membership.
        return list(dict.fromkeys(codes)), complete, True


CODECS = {'semantic': SemanticSIDCodec, 'random': RandomSIDCodec,
          'atomic': AtomicIDCodec, 'raw': RawCodec}


def make_codec(tokenizer, bundle):
    return CODECS[bundle['manifest'].get('representation', 'semantic')](tokenizer, bundle)


def representation_text(bundle, code):
    representation = bundle['manifest'].get('representation', 'semantic')
    values = bundle['assignments'][code]
    if representation == 'raw':
        return code
    if representation == 'atomic':
        return f'<ATOM_{values[0]:05d}>'
    return ' '.join(code_tokens(values))


# Existing SemICD consumers retain their historical class name.
Codec = SemanticSIDCodec
