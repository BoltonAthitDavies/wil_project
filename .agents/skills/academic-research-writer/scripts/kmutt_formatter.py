#!/usr/bin/env python3
"""Format reference-list entries in the KMUTT Faculty of Engineering style.

The style comes from "6. การเขียนเอกสารอ้างอิง" and the English preparation
manual in docs/MyThesisInfo/27.8.67 Template เล่มโครงงาน (Eng Ver.).  Its shape
is: author, YEAR SECOND, title, bold source, Vol./No., pp.  It is not IEEE.

Plain text output marks the bold source with **...**; --latex emits the
\\bibitem form used by docs/report/final_report/chapters/references.tex.

    python3 kmutt_formatter.py journal \\
        --authors "Yeh, T.S." "Sacks, M.D." --year 1988 \\
        --title "Effect of Particle Size Distribution on the Sintering of Alumina" \\
        --source "Journal of the American Ceramic Society" \\
        --volume 71 --number 12 --pages 484-487

    Yeh, T.S. and Sacks, M.D., 1988, "Effect of Particle Size Distribution on
    the Sintering of Alumina," **Journal of the American Ceramic Society**,
    Vol. 71, No. 12, pp. 484-487.

Run with no arguments for a worked example of every entry type.
"""

import argparse
import re
import sys
import unicodedata

# ---------------------------------------------------------------- authors

AUTHOR_RE = re.compile(r"^[^,]+,\s*(?:[A-Z]\.){1,4}$")


def check_author(name):
    """Warn when a name is not 'Surname, I.I.' as the manual sets it."""
    if not AUTHOR_RE.match(name.strip()):
        print(
            "warning: %r is not 'Surname, I.I.' (initials run together, "
            "each with a period)" % name,
            file=sys.stderr,
        )
    return name.strip()


def join_authors(authors, editors=False):
    """'A, B and C' -- commas between, 'and' before the last."""
    names = [check_author(a) for a in authors]
    if len(names) == 1:
        joined = names[0]
    else:
        joined = ", ".join(names[:-1]) + " and " + names[-1]
    if editors:
        joined += " (Eds.)" if len(names) > 1 else " (Ed.)"
    return joined


# ---------------------------------------------------------------- pieces


def quoted(title, latex):
    """Article title in quotes with the comma INSIDE, per the English edition."""
    title = title.rstrip().rstrip(",")
    return "``%s,''" % title if latex else '"%s,"' % title


def bold(text, latex):
    return "\\textbf{%s}" % text if latex else "**%s**" % text


def pages(spec, latex):
    if not spec:
        return None
    spec = str(spec).replace("--", "-")
    if latex:
        spec = spec.replace("-", "--")
        return "pp.~%s" % spec
    return "pp. %s" % spec


def volno(volume, number, latex):
    out = []
    nbsp = "~" if latex else " "
    if volume:
        out.append("Vol.%s%s" % (nbsp, volume))
    if number:
        out.append("No.%s%s" % (nbsp, number))
    return out


def assemble(parts):
    """Join the non-empty parts with commas and close with a full stop."""
    body = ", ".join(p for p in parts if p)
    # A quoted article title already carries its comma inside the closing
    # quotes, so the join must not add a second one: '"Title," ,' -> '"Title,"'
    body = re.sub(r",(\"|\'\')\s*,\s*", r",\1 ", body)
    body = re.sub(r",\s*,", ",", body)
    return body if body.endswith(".") else body + "."


def to_latex_accents(text):
    """pdflatex builds this volume, so write accents as control sequences."""
    simple = {
        "\u00e1": "\\'a", "\u00e9": "\\'e", "\u00ed": "\\'i", "\u00f3": "\\'o",
        "\u00fa": "\\'u", "\u00c1": "\\'A", "\u00c9": "\\'E", "\u00cd": "\\'I",
        "\u00d3": "\\'O", "\u00da": "\\'U", "\u00e0": "\\`a", "\u00e8": "\\`e",
        "\u00f1": "\\~n", "\u00e4": '\\"a', "\u00f6": '\\"o', "\u00fc": '\\"u',
        "\u00e7": "\\c{c}", "\u00e2": "\\^a", "\u00ea": "\\^e", "\u00f4": "\\^o",
        "\u00e5": "\\aa{}", "\u00f8": "\\o{}", "\u00df": "\\ss{}",
    }
    out = []
    for ch in text:
        if ch in simple:
            out.append(simple[ch])
        elif ord(ch) < 128:
            out.append(ch)
        else:
            decomposed = unicodedata.normalize("NFD", ch)
            out.append(decomposed[0] if decomposed else ch)
    return "".join(out)


# ---------------------------------------------------------------- entries


def journal(a, latex):
    return assemble([join_authors(a.authors), a.year, quoted(a.title, latex),
                     bold(a.source, latex), *volno(a.volume, a.number, latex),
                     pages(a.pages, latex)])


def conference(a, latex):
    return assemble([join_authors(a.authors), a.year, quoted(a.title, latex),
                     bold(a.source, latex), a.date, a.place,
                     pages(a.pages, latex)])


def book(a, latex):
    return assemble([join_authors(a.authors, editors=a.editors), a.year,
                     bold(a.title, latex), a.edition, a.publisher, a.place,
                     pages(a.pages, latex)])


def inbook(a, latex):
    return assemble([join_authors(a.authors), a.year, quoted(a.title, latex),
                     "In " + bold(a.source, latex),
                     (join_authors(a.editors_names, editors=True)
                      if a.editors_names else None),
                     a.edition, a.publisher, a.place, pages(a.pages, latex)])


def thesis(a, latex):
    return assemble([join_authors(a.authors), a.year, bold(a.title, latex),
                     a.degree, a.major, a.faculty, a.university,
                     pages(a.pages, latex)])


def newspaper(a, latex):
    return assemble([join_authors(a.authors), a.year, quoted(a.title, latex),
                     bold(a.source, latex), a.date, pages(a.pages, latex)])


def patent(a, latex):
    return assemble([a.owner, a.year, bold(a.title, latex), a.country,
                     a.number])


def online(a, latex):
    url = "\\url{%s}" % a.url if latex else a.url
    tail = "Available : %s [%s]" % (url, a.accessed)
    head = [join_authors(a.authors) if a.authors else None, a.year]
    if a.source:                      # article reached through a database
        return assemble([*head, quoted(a.title, latex),
                         bold(a.source, latex) + " [Electronic]",
                         *volno(a.volume, a.number, latex),
                         pages(a.pages, latex), tail])
    return assemble([*head, bold(a.title, latex) + " [Online]", tail])


BUILDERS = {"journal": journal, "conference": conference, "book": book,
            "inbook": inbook, "thesis": thesis, "newspaper": newspaper,
            "patent": patent, "online": online}


def citation_key(args):
    if getattr(args, "key", None):
        return args.key
    first = (args.authors[0] if getattr(args, "authors", None)
             else getattr(args, "owner", "anon"))
    surname = re.split(r"[,\s]", first.strip())[0].lower()
    surname = "".join(c for c in to_latex_accents(surname) if c.isalnum())
    return "%s%s" % (surname, args.year or "")


# ---------------------------------------------------------------- CLI


def build_parser():
    p = argparse.ArgumentParser(
        description="Format a reference in the KMUTT Faculty of Engineering "
                    "style (not IEEE).",
        epilog="See references/KMUTT-CITATION-GUIDE.md for every entry type.")
    p.add_argument("--latex", action="store_true",
                   help="emit the \\bibitem form for references.tex")
    sub = p.add_subparsers(dest="kind")

    def common(sp, authors=True):
        if authors:
            sp.add_argument("--authors", nargs="+", required=True,
                            metavar="'Surname, I.I.'")
        sp.add_argument("--year", required=True)
        sp.add_argument("--title", required=True)
        sp.add_argument("--pages")
        sp.add_argument("--key", help="citation key (default: surname+year)")

    sp = sub.add_parser("journal"); common(sp)
    sp.add_argument("--source", required=True, metavar="JOURNAL")
    sp.add_argument("--volume"); sp.add_argument("--number")

    sp = sub.add_parser("conference"); common(sp)
    sp.add_argument("--source", required=True, metavar="CONFERENCE")
    sp.add_argument("--date"); sp.add_argument("--place")

    sp = sub.add_parser("book"); common(sp)
    sp.add_argument("--edition"); sp.add_argument("--publisher")
    sp.add_argument("--place")
    sp.add_argument("--editors", action="store_true",
                    help="the names given are editors, not authors")

    sp = sub.add_parser("inbook"); common(sp)
    sp.add_argument("--source", required=True, metavar="BOOK")
    sp.add_argument("--editors-names", nargs="*", dest="editors_names")
    sp.add_argument("--edition"); sp.add_argument("--publisher")
    sp.add_argument("--place")

    sp = sub.add_parser("thesis"); common(sp)
    sp.add_argument("--degree", required=True)
    sp.add_argument("--major"); sp.add_argument("--faculty")
    sp.add_argument("--university")

    sp = sub.add_parser("newspaper"); common(sp)
    sp.add_argument("--source", required=True, metavar="NEWSPAPER")
    sp.add_argument("--date")

    sp = sub.add_parser("patent")
    sp.add_argument("--owner", required=True)
    sp.add_argument("--year", required=True)
    sp.add_argument("--title", required=True)
    sp.add_argument("--country"); sp.add_argument("--number")
    sp.add_argument("--key")

    sp = sub.add_parser("online")
    sp.add_argument("--authors", nargs="*", default=[])
    sp.add_argument("--year")
    sp.add_argument("--title", required=True)
    sp.add_argument("--url", required=True)
    sp.add_argument("--accessed", required=True,
                    metavar="'2026, October 22'",
                    help="access date, YEAR FIRST as the manual sets it")
    sp.add_argument("--source", help="journal, for a database article")
    sp.add_argument("--volume"); sp.add_argument("--number")
    sp.add_argument("--pages"); sp.add_argument("--key")
    return p


def render(args):
    entry = BUILDERS[args.kind](args, args.latex)
    if not args.latex:
        return entry
    return "\\bibitem{%s}\n%s" % (citation_key(args), to_latex_accents(entry))


def demo():
    p = build_parser()
    examples = [
        ["journal", "--authors", "Yeh, T.S.", "Sacks, M.D.", "--year", "1988",
         "--title", "Effect of Particle Size Distribution on the Sintering of "
         "Alumina", "--source", "Journal of the American Ceramic Society",
         "--volume", "71", "--number", "12", "--pages", "484-487"],
        ["conference", "--authors", "Merin, U.", "Daufin, G.", "--year", "1989",
         "--title", "Separation Process Using Inorganic Membrane in the Food "
         "Industry", "--source", "International Conference on Inorganic "
         "Membranes", "--date", "6 July 1989", "--place", "Paris",
         "--pages", "272-278"],
        ["book", "--authors", "Smiths, J.E.", "--year", "1990",
         "--title", "Engineer", "--edition", "2nd ed.",
         "--publisher", "Amalin.", "--place", "Bangkok.",
         "--pages", "120-150"],
        ["inbook", "--authors", "Waechter, E.H.", "--year", "1987",
         "--title", "How Families Cope : Accessing and Intervening",
         "--source", "The Child and Family Facing Life-threatening Illness",
         "--editors-names", "Krulik, T.", "--publisher", "Lippincott",
         "--place", "Philadelphia", "--pages", "239-242"],
        ["thesis", "--authors", "Mettripun, N.", "--year", "2005",
         "--title", "Reduced VIC Code Length for Very Low Bit Rate Video "
         "Coding", "--degree", "Master of Engineering Thesis",
         "--major", "Computer Engineering", "--faculty", "Faculty of "
         "Engineering", "--university", "King Mongkut's University of "
         "Technology Thonburi", "--pages", "1-59"],
        ["newspaper", "--authors", "Mena, N.", "--year", "2014",
         "--title", "Ceramic Society of Japan", "--source", "Daily News",
         "--date", "May 6.", "--pages", "3"],
        ["patent", "--owner", "Pinne, S.", "--year", "2009",
         "--title", "Chemi Model", "--country", "Paris",
         "--number", "3421.352.110"],
        ["online", "--authors", "Wu, K.", "--title", "What is Nano",
         "--url", "http://www.nano.org.uk/nano.htm",
         "--accessed", "2001, October 5"],
    ]
    print("KMUTT reference formats (author, YEAR SECOND, title, bold source)\n")
    for argv in examples:
        args = p.parse_args(argv)
        print("%s:" % args.kind)
        print("  " + render(args).replace("\n", "\n  "))
        args = p.parse_args(["--latex"] + argv)
        print("  " + render(args).replace("\n", "\n  "))
        print()


def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        demo()
        return 0
    args = parser.parse_args()
    if not args.kind:
        parser.print_help()
        return 2
    print(render(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
