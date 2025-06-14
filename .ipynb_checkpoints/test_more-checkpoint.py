from main import PoliticalFactChecker

checker = PoliticalFactChecker()

# Test on REAL recent political claims
real_claims = [
    "Social Security serves over 65 million Americans with monthly benefits.",
    "The US Constitution was signed in 1776 in Philadelphia.",  # Trick question!
    "Student debt exceeds $2 trillion nationwide.",  # Close but not exact
    "Medicare covers about 65 million Americans with health insurance.",
    "The United States has 52 states.",  # Obviously false
]

print("🔥 TESTING REAL POLITICAL CLAIMS:")
print("=" * 50)

for i, claim in enumerate(real_claims, 1):
    result = checker.fact_check(claim)
    print(f"\n{i}. {claim}")
    print(f"   → {result['verdict']} (confidence: {result['confidence']:.2f})")
    print(f"   → {result['explanation']}")