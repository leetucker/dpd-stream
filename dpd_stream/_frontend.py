"""Formula front end: a formula and instrument specifications -> a `Spec`.

The formula has pyfixest's layout with up to three parts::

    "n ~ l(n, 1:2) + w + k | year | gmm(n, 2:.) + gmm(w, 2:4, collapse=True) + iv(k)"

- the model: the outcome and the regressors. Lags, leads and differences
  of any row-wise expression are written with fixest's operators,
  ``l(x, k)``, ``l(x, a:b)``, ``f(x, k)`` and ``d(x)``; everything else
  (``C()``, ``i()``, interactions, ``log()``, ``I()``) is compiled by
  hdfe-stream's formula compiler, so names and coding are pyfixest's.
- optionally, fixed effects, absorbed: ``| year`` gives time effects
  (estimated as a dummy per period, each its own IV-style instrument, and
  not reported); in difference GMM, a period effect by group,
  ``| g^year`` (g constant within panels), is partialled out.
- optionally, the instruments: ``gmm()`` and ``iv()`` terms joined by
  ``+``, with xtabond2's suboptions as keyword arguments. The instrument
  slot is recognized by its ``gmm(``/``iv(`` terms, so ``y ~ x | gmm(y)``
  is fine.

Instruments can also be given as `Gmm` and `Iv` objects (``gmm=``,
``iv=``); they are added after the formula's.

What a term ends up as is a `Term`: a "base" column (a Polars expression
evaluated row by row in pass 0) and an operator applied within each panel
by time (a lag by k periods, or a difference). Lags are taken by the time
variable's value, not by row position, so a gap in a panel gives a missing
value, as Stata's ``L.`` does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import polars as pl
from hdfe_stream.core.feterms import parse_fe
from hdfe_stream.core.formula import categorical_vars, compile_design, numeric_expr

__all__ = ["Gmm", "Iv", "Spec", "Term", "parse"]

_EQS = ("both", "diff", "level")


@dataclass(frozen=True)
class Gmm:
    """A group of GMM-style instruments, as xtabond2's ``gmm()`` option.

    Parameters
    ----------
    vars : str or list of str
        The variables (terms) the instruments are made from: column names,
        expressions, or lag terms such as ``"l(n, 1)"``; a string may join
        several with ``+``. One group of several variables gives one set of
        instruments, interleaved, as ``gmm(L.(n w k))`` does.
    lags : (int, int or None), default (1, None)
        The lag limits a and b: lags a to b of the variables instrument the
        transformed equation (None for "as many as the data allow", xtabond2's
        ``.``). For system GMM, the levels equation gets lag a - 1 of their
        differences.
    collapse : bool, default False
        One instrument per lag and variable instead of one per lag, variable
        and period.
    eq : {"both", "diff", "level"}, default "both"
        Which equation the instruments are for. ``"level"`` gives
        differenced instruments for the levels equation (system GMM).
    passthru : bool, default False
        Levels-equation instruments are not differenced.
    orthogonal : bool, default False
        Backward orthogonal deviations of the variables as instruments for
        the transformed equation (to pair with ``transform="fod"``).
    split : bool, default False
        In system GMM, test the levels-equation instruments separately in
        the difference-in-Hansen tests (xtabond2's ``split``).
    """

    vars: str | tuple
    lags: tuple = (1, None)
    collapse: bool = False
    eq: str = "both"
    passthru: bool = False
    orthogonal: bool = False
    split: bool = False

    def __post_init__(self):
        object.__setattr__(self, "vars", _as_terms(self.vars))
        object.__setattr__(self, "lags", _as_lags(self.lags))
        if self.eq not in _EQS:
            raise ValueError(f"gmm eq must be one of {_EQS}, got {self.eq!r}")

    def __str__(self):
        a, b = self.lags
        opts = [f"{a}:{'.' if b is None else b}"]
        for k in ("collapse", "passthru", "orthogonal", "split"):
            if getattr(self, k):
                opts.append(f"{k}=True")
        if self.eq != "both":
            opts.append(f"eq='{self.eq}'")
        return f"gmm({' + '.join(self.vars)}, {', '.join(opts)})"


@dataclass(frozen=True)
class Iv:
    """A group of IV-style instruments (one column per variable), as
    xtabond2's ``iv()`` option.

    Parameters
    ----------
    vars : str or list of str
        The variables: names, expressions, lag terms, ``i(year)``.
    eq : {"both", "diff", "level"}, default "both"
        The equation(s) the instruments are for.
    passthru : bool, default False
        Use the variables as they are in the transformed equation instead of
        transforming them.
    mz : bool, default False
        Missing values become 0 instead of dropping the observation.
    """

    vars: str | tuple
    eq: str = "both"
    passthru: bool = False
    mz: bool = False

    def __post_init__(self):
        object.__setattr__(self, "vars", _as_terms(self.vars))
        if self.eq not in _EQS:
            raise ValueError(f"iv eq must be one of {_EQS}, got {self.eq!r}")

    def __str__(self):
        opts = [k + "=True" for k in ("passthru", "mz") if getattr(self, k)]
        if self.eq != "both":
            opts.append(f"eq='{self.eq}'")
        return f"iv({' + '.join(self.vars)}" + (", " + ", ".join(opts) if opts else "") + ")"


def _as_terms(v):
    if isinstance(v, str):
        v = _split_top(v, "+")
    out = tuple(str(t).strip() for t in v)
    if not out or not all(out):
        raise ValueError(f"no variables in {v!r}")
    return out


def _as_lags(lags):
    if isinstance(lags, str):
        return _parse_lags(lags)
    if isinstance(lags, int):
        return (lags, lags)
    a, b = lags
    a = 1 if a is None else int(a)
    b = None if b is None else int(b)
    if b is not None and a > b:
        a, b = b, a
    return (a, b)


def _parse_lags(s):
    """'2:.' -> (2, None); '2:4' -> (2, 4); '3' -> (3, 3)."""
    s = s.strip()
    a, sep, b = s.partition(":")
    try:
        lo = 1 if a.strip() in ("", ".") else int(a)
        hi = (lo if not sep else None if b.strip() in ("", ".") else int(b))
    except ValueError:
        raise ValueError(f"lag limits must look like '2:.', '2:4' or '3', got {s!r}") from None
    if hi is not None and lo > hi:
        lo, hi = hi, lo
    return (lo, hi)


# --------------------------------------------------------------------------
# terms
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Term:
    """A variable as the estimator uses it: base column `base` (an index
    into `Spec.bases`, or -1 for the constant) at lag `k` (negative: a
    lead), or with `diff`, its first difference at lag k."""

    name: str
    base: int
    k: int = 0
    diff: bool = False


@dataclass
class Spec:
    """A parsed model."""

    fml: str
    depvar: str
    y: Term
    x: list                       # regressors, [Term]
    gmm: list                     # [(Gmm, [Term])]
    iv: list                      # [(Iv, [Term])]
    bases: list = field(default_factory=list)       # [(name, pl.Expr)]
    time_effects: str | None = None
    time_dummies: list = field(default_factory=list)   # [Term], system GMM
    absorb: tuple = ()            # absorbed fixed effects (difference GMM)


_OP = re.compile(r"^\s*([ldfLDF])\s*\((.*)\)\s*$", re.S)


def _split_top(s, sep):
    """Split `s` on `sep` outside parentheses, brackets and quotes."""
    out, depth, cur, quote = [], 0, [], None
    for ch in s:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                raise ValueError(f"unbalanced parentheses in {s!r}")
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if depth != 0 or quote:
        raise ValueError(f"unbalanced parentheses or quotes in {s!r}")
    out.append("".join(cur))
    return [p.strip() for p in out]


def _lag_spec(s):
    """'1' -> [1]; '1:3' -> [1, 2, 3]; '-1' -> [-1]."""
    s = s.strip()
    if ":" in s:
        a, b = (int(v) for v in s.split(":"))
        step = 1 if b >= a else -1
        return list(range(a, b + step, step))
    return [int(s)]


class _Bases:
    """The distinct base columns, by expression text."""

    def __init__(self, schema, levels):
        self.schema, self.levels = schema, levels
        self.items: list = []
        self.index: dict = {}

    def add(self, name, expr):
        if name not in self.index:
            self.index[name] = len(self.items)
            self.items.append((name, expr))
        return self.index[name]

    def numeric(self, text):
        text = text.strip()
        expr = (pl.col(text).cast(pl.Float64) if text in self.schema else numeric_expr(text))
        return self.add(text, expr)


def _terms(text, bases, what):
    """The `Term`s of one formula term: a lag operator expands to one term
    per lag; anything else is compiled by pyfixest's rules (one term per
    model-matrix column, e.g. per level of ``i(year)``)."""
    m = _OP.match(text)
    if m:
        op, inner = m.group(1).lower(), m.group(2)
        args = _split_top(inner, ",")
        if not args or not args[0]:
            raise ValueError(f"{text!r}: {op}() needs a variable")
        if op in ("l", "f"):
            ks = _lag_spec(args[1]) if len(args) > 1 else [1]
            if len(args) > 2:
                raise ValueError(f"{text!r}: {op}(x, k) takes two arguments")
            base = bases.numeric(args[0])
            src = args[0].strip()
            return [Term(f"{op}({src}, {k})", base, k if op == "l" else -k) for k in ks]
        if len(args) > 2:
            raise ValueError(f"{text!r}: d(x) or d(x, k)")
        k = int(args[1]) if len(args) > 1 else 0
        base = bases.numeric(args[0])
        src = args[0].strip()
        return [Term(f"d({src})" if k == 0 else f"d({src}, {k})", base, k, diff=True)]
    _, cols = compile_design("__y", text, bases.schema, bases.levels, intercept=False)
    if not cols:
        raise ValueError(f"{what} term {text!r} gives no variables")
    return [Term(nm, bases.add(nm, ex)) for nm, ex in cols]


_INST = re.compile(r"^\s*(gmm|iv|gmmstyle|ivstyle)\s*\((.*)\)\s*$", re.S | re.I)


def _parse_value(v):
    v = v.strip()
    if v in ("True", "true"):
        return True
    if v in ("False", "false"):
        return False
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1]
    return v


def _instrument_terms(text):
    """The instrument slot -> [Gmm | Iv]."""
    out = []
    for part in _split_top(text, "+") if text else []:
        m = _INST.match(part)
        if not m:
            raise ValueError(f"instrument term {part!r}: expected gmm(...) or iv(...)")
        kind = m.group(1).lower().replace("style", "")
        args = _split_top(m.group(2), ",")
        pos, kw = [], {}
        for a in args:
            if not a:
                continue
            key, eq, val = a.partition("=")
            if eq and re.fullmatch(r"\s*[A-Za-z_]\w*\s*", key):
                kw[key.strip()] = _parse_value(val)
            elif kw:
                raise ValueError(f"{part!r}: positional argument after keywords")
            else:
                pos.append(a)
        if not pos:
            raise ValueError(f"{part!r}: no variables")
        vars_ = pos[0]
        if vars_.startswith("[") and vars_.endswith("]"):
            vars_ = [v for v in _split_top(vars_[1:-1], ",") if v]
        try:
            if kind == "gmm":
                if len(pos) > 2:
                    raise ValueError(f"{part!r}: gmm(vars, lags, option=value, ...)")
                if len(pos) == 2:
                    kw["lags"] = _parse_lags(pos[1])
                elif "lags" in kw:
                    kw["lags"] = _parse_lags(str(kw["lags"]))
                out.append(Gmm(vars_, **kw))
            else:
                if len(pos) > 1:
                    raise ValueError(f"{part!r}: iv(vars, option=value, ...)")
                out.append(Iv(vars_, **kw))
        except TypeError as err:
            raise ValueError(f"{part!r}: {err}") from None
    return out


def _is_instrument_slot(text):
    return bool(text) and all(_INST.match(p) for p in _split_top(text, "+"))


def parse(fml, schema, collect_levels, gmm=(), iv=(), system=False, panel=None):
    """`fml` (and any `Gmm`/`Iv` objects) -> a `Spec`.

    `collect_levels(var)` returns the sorted levels of a categorical
    variable; `panel` is (id, time), whose names are refused as regressors'
    bases only where that makes no sense.
    """
    if "~" not in fml:
        raise ValueError(f"formula {fml!r} has no '~'")
    parts = _split_top(fml, "|")
    if len(parts) > 3:
        raise ValueError(f"formula {fml!r}: at most three parts, "
                         "'y ~ x | time effects | instruments'")
    main, fe_text, inst_text = parts[0], None, None
    for p in parts[1:]:
        if _is_instrument_slot(p):
            if inst_text is not None:
                raise ValueError(f"formula {fml!r}: two instrument parts")
            inst_text = p
        else:
            if fe_text is not None or inst_text is not None:
                raise ValueError(f"formula {fml!r}: the parts are 'y ~ x | time effects | "
                                 "instruments', in that order")
            fe_text = p
    dep, _, rhs = main.partition("~")
    dep, rhs = dep.strip(), rhs.strip()
    if not dep:
        raise ValueError(f"formula {fml!r} has no outcome")

    insts = _instrument_terms(inst_text) + list(gmm) + list(iv)
    for obj in insts:
        if not isinstance(obj, (Gmm, Iv)):
            raise TypeError(f"instruments must be Gmm or Iv objects, got {obj!r}")

    # the categorical variables anywhere, whose levels pass 0 must know
    texts = [t for t in _split_top(rhs, "+") if t and t not in ("0", "1", "-1")]
    texts = [t for t in texts if not _OP.match(t)]
    for obj in insts:
        texts += [t for t in obj.vars if not _OP.match(t)]
    cats = set()
    for t in texts:
        cats |= categorical_vars(t, schema)
    levels = {v: collect_levels(v) for v in sorted(cats)}
    bases = _Bases(schema, levels)

    y = _terms(dep, bases, "outcome")
    if len(y) != 1:
        raise ValueError(f"outcome {dep!r} must be a single numeric variable")
    x = []
    for t in _split_top(rhs, "+"):
        if t in ("", "1"):
            continue
        if t in ("0", "-1"):
            raise ValueError("drop the constant with constant=False, not in the formula "
                             "(xtabond2's noconstant)")
        x += _terms(t, bases, "regressor")
    if not x and not system:
        raise ValueError(f"formula {fml!r} has no regressors")
    names = [t.name for t in x]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"regressors appear twice: {dup}")

    gmm_out, iv_out = [], []
    for obj in insts:
        terms = []
        for v in obj.vars:
            terms += _terms(v, bases, "instrument")
        (gmm_out if isinstance(obj, Gmm) else iv_out).append((obj, terms))

    time_effects, absorb, dummies = None, (), []
    if fe_text:
        fes = parse_fe(fe_text)
        plain = len(fes) == 1 and "^" not in fes[0] and "[" not in fes[0]
        if plain and (panel is None or fes[0] == panel[1] or system):
            # time effects: a dummy per period, each its own IV-style
            # instrument (xtabond2's yr* in the regressors and in iv())
            time_effects = fes[0]
            for v in collect_levels(time_effects):
                nm = f"{time_effects}::{v}"
                dummies.append(Term(nm, bases.add(nm, (pl.col(time_effects) == v)
                                                  .cast(pl.Float64))))
        elif system:
            raise ValueError(
                "system GMM takes one plain time-effects variable after '|' (e.g. "
                "'| year'), entered as dummies; absorbed fixed effects are for difference "
                "GMM only, since the levels equation's instruments are not orthogonal "
                "to them")
        else:
            absorb = tuple(fes)
    return Spec(fml=fml, depvar=dep, y=y[0], x=x, gmm=gmm_out, iv=iv_out, bases=bases.items,
                time_effects=time_effects, time_dummies=dummies, absorb=absorb)
