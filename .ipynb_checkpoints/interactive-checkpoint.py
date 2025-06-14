from main import PoliticalFactChecker

def interactive_session():
    checker = PoliticalFactChecker()
    
    print("🎯 INTERACTIVE POLITICAL FACT-CHECKER")
    print("Enter political statements to fact-check (type 'quit' to exit)")
    print("="*60)
    
    while True:
        statement = input("\n📝 Enter statement: ").strip()
        if statement.lower() in ['quit', 'exit', 'q']:
            break
            
        result = checker.fact_check(statement)
        print(f"\n🏆 VERDICT: {result['verdict']}")
        print(f"🎯 CONFIDENCE: {result['confidence']:.2f}")
        print(f"💡 EXPLANATION: {result['explanation']}")

if __name__ == "__main__":
    interactive_session()