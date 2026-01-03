"""
Literature Search Helper for τ = 27/10 Formula
==============================================

Tool to help systematically search for precedent in string theory literature.

This script:
1. Generates search queries for ArXiv
2. Provides reading checklist for key papers
3. Tracks findings and citations
4. Assesses novelty

Author: Kevin Heitfeld
Date: December 28, 2025
"""

import json
from datetime import datetime
from typing import List, Dict

# ============================================================================
# SEARCH QUERIES
# ============================================================================

def generate_arxiv_queries() -> List[Dict[str, str]]:
    """
    Generate systematic ArXiv search queries.

    Returns list of search queries with metadata.
    """
    queries = [
        {
            "id": "Q1",
            "name": "Direct Formula Search",
            "query": 'abs:"complex structure" AND abs:"orbifold" AND (abs:"formula" OR abs:"determination")',
            "category": "hep-th",
            "expected_results": "50-100",
            "priority": "HIGH"
        },
        {
            "id": "Q2",
            "name": "Modular Parameter from Orbifold",
            "query": 'abs:"modular parameter" AND (abs:"Z_3" OR abs:"Z_4" OR abs:"orbifold") AND abs:"compactification"',
            "category": "hep-th",
            "expected_results": "20-50",
            "priority": "HIGH"
        },
        {
            "id": "Q3",
            "name": "Rational Tau Values",
            "query": 'abs:"rational" AND abs:"tau" AND abs:"string" AND abs:"modular"',
            "category": "hep-th, math-ph",
            "expected_results": "30-60",
            "priority": "MEDIUM"
        },
        {
            "id": "Q4",
            "name": "Product Orbifold Z3xZ4",
            "query": '(abs:"Z_3 times Z_4" OR abs:"Z3 x Z4" OR abs:"Z_3 x Z_4") AND abs:"complex structure"',
            "category": "hep-th",
            "expected_results": "5-15",
            "priority": "HIGH"
        },
        {
            "id": "Q5",
            "name": "Hodge Numbers and Moduli",
            "query": 'abs:"h^{1,1}" AND abs:"complex structure" AND abs:"orbifold"',
            "category": "hep-th",
            "expected_results": "40-80",
            "priority": "MEDIUM"
        },
        {
            "id": "Q6",
            "name": "Type IIB Moduli Determination",
            "query": 'abs:"Type IIB" AND abs:"complex structure" AND (abs:"stabilization" OR abs:"determination")',
            "category": "hep-th",
            "expected_results": "100-200",
            "priority": "MEDIUM"
        }
    ]

    return queries


def print_arxiv_instructions():
    """
    Print instructions for using ArXiv search.
    """
    print("=" * 70)
    print("ArXiv SEARCH INSTRUCTIONS")
    print("=" * 70)
    print()
    print("How to search ArXiv effectively:")
    print()
    print("1. Go to: https://arxiv.org/search/advanced")
    print()
    print("2. For each query below:")
    print("   - Copy the query string")
    print("   - Paste into 'Abstract' field")
    print("   - Select category (hep-th, hep-ph, etc.)")
    print("   - Set date range if needed (e.g., last 10 years)")
    print("   - Click 'Search'")
    print()
    print("3. Scan results:")
    print("   - Read titles and abstracts")
    print("   - Download 5-10 most relevant papers")
    print("   - Mark any exact formula matches")
    print()
    print("4. Record findings in TAU_LITERATURE_SEARCH.md")
    print()

    queries = generate_arxiv_queries()

    for q in queries:
        print("-" * 70)
        print(f"QUERY {q['id']}: {q['name']} [{q['priority']} priority]")
        print(f"Category: {q['category']}")
        print(f"Expected: {q['expected_results']} results")
        print()
        print("Search query:")
        print(f"  {q['query']}")
        print()
        print("ArXiv URL format:")
        # Encode query for URL
        import urllib.parse
        encoded = urllib.parse.quote(q['query'])
        print(f"  https://arxiv.org/search/?query={encoded}&searchtype=abstract")
        print()


# ============================================================================
# KEY PAPERS CHECKLIST
# ============================================================================

def generate_papers_checklist() -> List[Dict[str, any]]:
    """
    Generate checklist of must-read papers.
    """
    papers = [
        {
            "category": "Modular Flavor (Recent)",
            "papers": [
                {
                    "arxiv": "2001.07972",
                    "title": "Classification of discrete modular symmetries in Type IIB flux vacua",
                    "authors": "Kobayashi-Otsuka",
                    "year": 2020,
                    "relevance": "Direct on modular symmetry from branes",
                    "sections_to_check": ["Sec 2: Moduli space", "Sec 3: Symmetries"],
                    "checked": False
                },
                {
                    "arxiv": "2408.13984",
                    "title": "Non-invertible flavor symmetries in magnetized extra dimensions",
                    "authors": "Kobayashi-Otsuka et al.",
                    "year": 2024,
                    "relevance": "Most recent, may have formula",
                    "sections_to_check": ["All sections on moduli"],
                    "checked": False
                }
            ]
        },
        {
            "category": "Magnetized Branes (Foundational)",
            "papers": [
                {
                    "arxiv": "hep-th/0308001",
                    "title": "Computing Yukawa couplings from magnetized D-branes",
                    "authors": "Cremades-Ibanez-Marchesano",
                    "year": 2003,
                    "relevance": "Wave functions, modular transformations",
                    "sections_to_check": ["Sec 3: Wave functions", "Sec 4: Yukawa"],
                    "checked": False
                },
                {
                    "arxiv": "hep-th/0607160",
                    "title": "Yukawa couplings in intersecting D-brane models",
                    "authors": "Ibanez-Marchesano-Rabadan",
                    "year": 2006,
                    "relevance": "Complex structure in Yukawa calculation",
                    "sections_to_check": ["Sec 2: Setup", "Appendix on moduli"],
                    "checked": False
                }
            ]
        },
        {
            "category": "Orbifold Classics",
            "papers": [
                {
                    "arxiv": "Nucl.Phys.B261:678",
                    "title": "Strings on Orbifolds",
                    "authors": "Dixon-Harvey-Vafa-Witten",
                    "year": 1985,
                    "relevance": "Original orbifold paper - foundational",
                    "sections_to_check": ["Sec 3: Moduli space", "Sec 4: Twisted sectors"],
                    "checked": False
                },
                {
                    "arxiv": "Nucl.Phys.B329:27",
                    "title": "Moduli dependence of string loop corrections",
                    "authors": "Dixon-Kaplunovsky-Louis",
                    "year": 1990,
                    "relevance": "Moduli in orbifold CFT",
                    "sections_to_check": ["Sec 2: Moduli space structure"],
                    "checked": False
                }
            ]
        },
        {
            "category": "F-theory / CY Moduli",
            "papers": [
                {
                    "arxiv": "1806.01854",
                    "title": "Lectures on F-theory compactifications and model building",
                    "authors": "Weigand",
                    "year": 2018,
                    "relevance": "Comprehensive review, complex structure discussion",
                    "sections_to_check": ["Sec 3: Complex structure", "Sec 5: Rational points"],
                    "checked": False
                },
                {
                    "arxiv": "hep-th/9611137",
                    "title": "Enhanced gauge symmetries and K3 surfaces",
                    "authors": "Aspinwall",
                    "year": 1996,
                    "relevance": "Special points in moduli space",
                    "sections_to_check": ["Sec 2: Moduli space geometry"],
                    "checked": False
                }
            ]
        }
    ]

    return papers


def print_papers_checklist():
    """
    Print reading checklist for key papers.
    """
    print("\n" + "=" * 70)
    print("KEY PAPERS READING CHECKLIST")
    print("=" * 70)
    print()
    print("Priority papers to check for formula τ = k/(N₁+N₂+h^{1,1})")
    print()

    papers = generate_papers_checklist()

    for category in papers:
        print(f"\n{category['category']}")
        print("-" * 70)

        for paper in category['papers']:
            status = "✓" if paper['checked'] else "☐"
            print(f"\n{status} [{paper['arxiv']}] {paper['title']}")
            print(f"   {paper['authors']} ({paper['year']})")
            print(f"   Relevance: {paper['relevance']}")
            print(f"   Check: {', '.join(paper['sections_to_check'])}")

    print("\n" + "=" * 70)
    print("READING STRATEGY")
    print("=" * 70)
    print()
    print("For each paper:")
    print("  1. Download PDF")
    print("  2. Read abstract and introduction")
    print("  3. Jump to sections listed above")
    print("  4. Search PDF for keywords: 'complex structure', 'formula', 'orbifold'")
    print("  5. Note any related formulas (even if not exact match)")
    print("  6. Mark as ✓ when done")
    print()


# ============================================================================
# TEXTBOOK SECTIONS
# ============================================================================

def print_textbook_guide():
    """
    Guide for checking standard textbooks.
    """
    print("\n" + "=" * 70)
    print("TEXTBOOK READING GUIDE")
    print("=" * 70)
    print()

    books = [
        {
            "title": "String Theory and Particle Physics",
            "authors": "Ibanez-Uranga",
            "status": "⏸ Need access",
            "chapters": [
                "Ch 10: Toroidal orientifolds and orbifolds",
                "Ch 11: Intersecting brane worlds",
                "Ch 12: Magnetized D-branes",
                "Appendix: Moduli space"
            ],
            "keywords": ["complex structure", "orbifold", "Z_N", "modular parameter"]
        },
        {
            "title": "Basic Concepts of String Theory",
            "authors": "Blumenhagen-Lüst-Theisen",
            "status": "⏸ Need access",
            "chapters": [
                "Ch 10: Toroidal compactifications",
                "Ch 11: Orbifold compactifications"
            ],
            "keywords": ["complex structure", "twisted sectors", "moduli"]
        },
        {
            "title": "Lectures on F-theory",
            "authors": "Weigand",
            "status": "✓ Available (arXiv:1806.01854)",
            "chapters": [
                "Sec 3: Complex structure moduli space",
                "Sec 5: Rational points and singular loci"
            ],
            "keywords": ["complex structure", "rational curves", "Type IIB limit"]
        }
    ]

    for book in books:
        print(f"\n{book['status']} {book['title']}")
        print(f"   by {book['authors']}")
        print(f"   Chapters to check:")
        for ch in book['chapters']:
            print(f"     • {ch}")
        print(f"   Search for: {', '.join(book['keywords'])}")

    print()


# ============================================================================
# FINDINGS TRACKER
# ============================================================================

class FindingsTracker:
    """
    Track literature search findings.
    """

    def __init__(self, filename='research/literature_findings.json'):
        self.filename = filename
        self.findings = []
        self.load()

    def load(self):
        """Load existing findings if file exists."""
        try:
            with open(self.filename, 'r') as f:
                data = json.load(f)
                self.findings = data.get('findings', [])
        except FileNotFoundError:
            self.findings = []

    def add_finding(self, source, content, assessment, exact_match=False):
        """
        Add a new finding.

        Parameters:
        -----------
        source : str
            Paper arxiv ID or book name
        content : str
            Relevant formula, quote, or summary
        assessment : str
            'EXACT_MATCH', 'RELATED', 'NOT_RELEVANT'
        exact_match : bool
            True if exact formula found
        """
        finding = {
            'date': datetime.now().isoformat(),
            'source': source,
            'content': content,
            'assessment': assessment,
            'exact_match': exact_match
        }

        self.findings.append(finding)
        self.save()

        if exact_match:
            print(f"\n🎯 EXACT MATCH FOUND in {source}!")
        else:
            print(f"\n📝 Finding recorded from {source}: {assessment}")

    def save(self):
        """Save findings to file."""
        data = {
            'search_date': datetime.now().isoformat(),
            'formula': 'τ = k_lepton / (N_Z3 + N_Z4 + h^{1,1})',
            'findings': self.findings
        }

        with open(self.filename, 'w') as f:
            json.dump(data, f, indent=2)

    def summary(self):
        """Print summary of findings."""
        print("\n" + "=" * 70)
        print("LITERATURE SEARCH SUMMARY")
        print("=" * 70)
        print()
        print(f"Total findings: {len(self.findings)}")

        exact = sum(1 for f in self.findings if f['exact_match'])
        related = sum(1 for f in self.findings if f['assessment'] == 'RELATED')
        not_rel = sum(1 for f in self.findings if f['assessment'] == 'NOT_RELEVANT')

        print(f"  Exact matches: {exact}")
        print(f"  Related formulas: {related}")
        print(f"  Not relevant: {not_rel}")
        print()

        if exact > 0:
            print("🎯 PRECEDENT FOUND!")
            print("\nExact matches:")
            for f in self.findings:
                if f['exact_match']:
                    print(f"  • {f['source']}: {f['content'][:100]}...")
        else:
            print("⚠️  No exact precedent found yet")
            if related > 0:
                print(f"\nFound {related} related formulas - may need deeper analysis")

        print()


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_search():
    """
    Interactive mode for recording findings.
    """
    tracker = FindingsTracker()

    print("\n" + "=" * 70)
    print("INTERACTIVE LITERATURE SEARCH")
    print("=" * 70)
    print()
    print("Commands:")
    print("  add   - Add a finding")
    print("  list  - List all findings")
    print("  sum   - Show summary")
    print("  quit  - Exit")
    print()

    while True:
        cmd = input("\nCommand: ").strip().lower()

        if cmd == 'quit':
            break
        elif cmd == 'add':
            source = input("  Source (arxiv/book): ")
            content = input("  Content/Formula: ")
            assessment = input("  Assessment (EXACT_MATCH/RELATED/NOT_RELEVANT): ").upper()
            exact = (assessment == 'EXACT_MATCH')

            tracker.add_finding(source, content, assessment, exact)

        elif cmd == 'list':
            for i, f in enumerate(tracker.findings, 1):
                print(f"\n{i}. [{f['source']}] - {f['assessment']}")
                print(f"   {f['content'][:100]}...")

        elif cmd == 'sum':
            tracker.summary()
        else:
            print("Unknown command. Use: add, list, sum, quit")

    tracker.summary()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main function - print all search aids.
    """
    print("\n" + "=" * 70)
    print("LITERATURE SEARCH HELPER")
    print("τ = k_lepton / (N_Z3 + N_Z4 + h^{1,1}) = 27/10")
    print("=" * 70)
    print()
    print("This tool helps you systematically search for precedent in the literature.")
    print()

    # Print all guides
    print_arxiv_instructions()
    print_papers_checklist()
    print_textbook_guide()

    # Summary
    print("\n" + "=" * 70)
    print("SEARCH WORKFLOW")
    print("=" * 70)
    print()
    print("Phase 1 (Today): Textbooks & ArXiv")
    print("  1. Read Weigand lectures (arXiv:1806.01854) - 2 hours")
    print("  2. Run ArXiv Query Q1 (direct formula) - 1 hour")
    print("  3. Download top 10 papers - 30 min")
    print("  4. Scan abstracts and conclusions - 30 min")
    print()
    print("Phase 2 (Tomorrow): Deep Dive")
    print("  5. Read key sections from top 5 papers - 3 hours")
    print("  6. Run remaining ArXiv queries - 1 hour")
    print("  7. Document all findings - 1 hour")
    print()
    print("Phase 3: Assessment")
    print("  8. Compile evidence for/against novelty")
    print("  9. Consult experts if unclear")
    print("  10. Make decision: Precedent OR Novel")
    print()
    print("To record findings interactively, run:")
    print("  python literature_search_helper.py --interactive")
    print()


if __name__ == "__main__":
    import sys

    if '--interactive' in sys.argv:
        interactive_search()
    else:
        main()
