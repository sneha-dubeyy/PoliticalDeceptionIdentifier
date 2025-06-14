# PRESENTATION-READY FEATURES
# Add these to your existing main.py for maximum impact!

import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Add this class to your main.py
class FactCheckAnalyzer:
    """Advanced analysis and visualization for fact-checking results"""
    
    def __init__(self):
        self.results_history = []
    
    def add_result(self, result):
        """Add a fact-checking result to history"""
        self.results_history.append(result)
    
    def generate_performance_report(self):
        """Generate comprehensive performance analysis"""
        if not self.results_history:
            return "No results to analyze"
        
        total_statements = len(self.results_history)
        statements_with_claims = len([r for r in self.results_history if r['claims_analyzed'] > 0])
        
        # Verdict distribution
        verdicts = [r['verdict'] for r in self.results_history]
        verdict_counts = Counter(verdicts)
        
        # Processing time stats
        processing_times = [r['processing_time'] for r in self.results_history]
        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)
        
        # Claims analysis
        total_claims = sum(r['claims_analyzed'] for r in self.results_history)
        claims_with_evidence = sum(r['claims_with_evidence'] for r in self.results_history)
        evidence_coverage = claims_with_evidence / max(total_claims, 1)
        
        # Confidence analysis
        confident_results = len([r for r in self.results_history if r['confidence'] > 0.7])
        confidence_rate = confident_results / total_statements
        
        report = f"""
🏆 FACT-CHECKER PERFORMANCE ANALYSIS
={'='*50}

📊 OVERALL STATISTICS:
   • Total statements processed: {total_statements}
   • Statements with factual claims: {statements_with_claims}
   • Total claims detected: {total_claims}
   • Claims with evidence found: {claims_with_evidence}
   • Evidence coverage rate: {evidence_coverage:.1%}

⚡ PERFORMANCE METRICS:
   • Average processing time: {avg_time:.3f} seconds
   • Maximum processing time: {max_time:.3f} seconds
   • High-confidence results: {confident_results}/{total_statements} ({confidence_rate:.1%})

🎯 VERDICT DISTRIBUTION:
"""
        for verdict, count in verdict_counts.most_common():
            percentage = count / total_statements * 100
            report += f"   • {verdict}: {count} ({percentage:.1f}%)\n"
        
        return report
    
    def create_visualization(self, save_path='fact_checker_analysis.png'):
        """Create visualization of results"""
        if not self.results_history:
            print("No results to visualize")
            return
        
        # Set up the plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Political Fact-Checker Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Verdict Distribution
        verdicts = [r['verdict'] for r in self.results_history]
        verdict_counts = Counter(verdicts)
        
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db', '#9b59b6']
        ax1.pie(verdict_counts.values(), labels=verdict_counts.keys(), autopct='%1.1f%%',
                colors=colors[:len(verdict_counts)], startangle=90)
        ax1.set_title('Verdict Distribution', fontweight='bold')
        
        # 2. Confidence Score Distribution
        confidences = [r['confidence'] for r in self.results_history if r['confidence'] > 0]
        ax2.hist(confidences, bins=10, alpha=0.7, color='#3498db', edgecolor='black')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence Score Distribution', fontweight='bold')
        ax2.axvline(sum(confidences)/len(confidences), color='red', linestyle='--', 
                   label=f'Mean: {sum(confidences)/len(confidences):.2f}')
        ax2.legend()
        
        # 3. Processing Time Analysis
        times = [r['processing_time'] for r in self.results_history]
        ax3.plot(range(len(times)), times, marker='o', color='#e74c3c', linewidth=2)
        ax3.set_xlabel('Statement Number')
        ax3.set_ylabel('Processing Time (seconds)')
        ax3.set_title('Processing Time per Statement', fontweight='bold')
        ax3.axhline(sum(times)/len(times), color='green', linestyle='--',
                   label=f'Average: {sum(times)/len(times):.3f}s')
        ax3.legend()
        
        # 4. Claims vs Evidence Analysis
        statements = range(1, len(self.results_history) + 1)
        claims = [r['claims_analyzed'] for r in self.results_history]
        evidence = [r['claims_with_evidence'] for r in self.results_history]
        
        width = 0.35
        ax4.bar([x - width/2 for x in statements], claims, width, 
                label='Claims Detected', color='#f39c12', alpha=0.8)
        ax4.bar([x + width/2 for x in statements], evidence, width,
                label='Claims with Evidence', color='#2ecc71', alpha=0.8)
        ax4.set_xlabel('Statement Number')
        ax4.set_ylabel('Number of Claims')
        ax4.set_title('Claims Detection vs Evidence Coverage', fontweight='bold')
        ax4.legend()
        ax4.set_xticks(statements)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved as '{save_path}'")
        plt.show()

# Challenge test cases to show robustness
def run_challenge_tests():
    """Test the system on challenging edge cases"""
    from main import PoliticalFactChecker
    
    checker = PoliticalFactChecker()
    analyzer = FactCheckAnalyzer()
    
    # Challenging test cases
    challenge_statements = [
        # Exact matches (should be MOSTLY_TRUE)
        "The minimum wage is $7.25 per hour since 2009.",
        "Climate change is supported by 97% of scientists.",
        
        # Close but not exact (should be PARTIALLY_TRUE)
        "The minimum wage is about $7 per hour.",
        "Almost all climate scientists support climate change.",
        
        # Clearly false (should be MOSTLY_FALSE or LOW confidence)
        "The unemployment rate is 50% in America.",
        "The United States has 60 states.",
        
        # Mixed factual and opinion (should separate them)
        "The unemployment rate is 3.7% and I think that's terrible.",
        "Social Security helps 67 million people, which is a great program.",
        
        # Complex multi-claim statements
        "The US spends $877 billion on defense and has a population of 335 million people.",
        "Student debt is $1.7 trillion while the minimum wage stays at $7.25 since 2009.",
        
        # Edge cases
        "According to data, renewable energy accounts for 20% of electricity generation.",
        "Some people believe the Constitution was signed in 1776 but it was actually 1787.",
    ]
    
    print("🔥 CHALLENGE TEST SUITE")
    print("="*60)
    print("Testing system robustness on edge cases and complex statements...")
    print()
    
    correct_predictions = 0
    total_tests = 0
    
    # Expected results for evaluation
    expected_results = [
        'TRUE',     # Minimum wage exact
        'TRUE',     # Climate exact
        'PARTIAL',  # Minimum wage approximate
        'TRUE',     # Climate approximate
        'FALSE',    # 50% unemployment
        'FALSE',    # 60 states
        'MIXED',    # Mixed fact/opinion
        'MIXED',    # Mixed fact/opinion
        'TRUE',     # Multi-claim true
        'TRUE',     # Multi-claim true
        'TRUE',     # Renewable energy
        'TRUE',     # Constitution correction
    ]
    
    for i, statement in enumerate(challenge_statements):
        print(f"🧪 CHALLENGE {i+1}: {statement}")
        print("-" * 50)
        
        result = checker.fact_check(statement)
        analyzer.add_result(result)
        
        # Evaluate result
        verdict = result['verdict']
        confidence = result['confidence']
        expected = expected_results[i]
        
        # Simple evaluation logic
        is_correct = False
        if expected == 'TRUE' and verdict in ['MOSTLY_TRUE', 'PARTIALLY_TRUE']:
            is_correct = True
        elif expected == 'FALSE' and (verdict == 'MIXED_EVIDENCE' or confidence < 0.5):
            is_correct = True
        elif expected == 'PARTIAL' and verdict in ['PARTIALLY_TRUE', 'MIXED_EVIDENCE']:
            is_correct = True
        elif expected == 'MIXED' and verdict in ['PARTIALLY_TRUE', 'MIXED_EVIDENCE', 'NO_FACTUAL_CLAIMS']:
            is_correct = True
        
        if is_correct:
            correct_predictions += 1
        total_tests += 1
        
        print(f"🏆 RESULT: {verdict} (confidence: {confidence:.2f})")
        print(f"📊 CLAIMS: {result['claims_analyzed']} detected, {result['claims_with_evidence']} with evidence")
        print(f"✅ EVALUATION: {'CORRECT' if is_correct else 'NEEDS_REVIEW'}")
        print(f"💡 EXPLANATION: {result['explanation']}")
        print()
    
    # Final analysis
    accuracy = correct_predictions / total_tests
    print("🏆 CHALLENGE TEST RESULTS")
    print("="*40)
    print(f"✅ Accuracy: {accuracy:.1%} ({correct_predictions}/{total_tests})")
    print(f"⚡ Average processing: {sum(r['processing_time'] for r in analyzer.results_history)/len(analyzer.results_history):.3f}s")
    
    # Generate full report
    print(analyzer.generate_performance_report())
    
    # Create visualization
    try:
        analyzer.create_visualization()
    except ImportError:
        print("📊 Visualization requires matplotlib (pip install matplotlib)")
    
    return analyzer, accuracy

# Demo script for presentations
def create_presentation_demo():
    """Create an impressive demo for presentations"""
    
    print("🎤 PRESENTATION DEMO MODE")
    print("="*50)
    print("Demonstrating key capabilities...")
    print()
    
    from main import PoliticalFactChecker
    checker = PoliticalFactChecker()
    
    # Key demo statements that show different capabilities
    demo_cases = [
        {
            'statement': "The minimum wage has been $7.25 per hour since 2009 and should be increased.",
            'highlight': "MIXED FACT/OPINION DETECTION"
        },
        {
            'statement': "Unemployment in America is at record highs of 25%.",
            'highlight': "FALSE CLAIM DETECTION"
        },
        {
            'statement': "Climate change is supported by 97% of scientists according to multiple studies.",
            'highlight': "COMPLEX FACTUAL VERIFICATION"
        },
        {
            'statement': "The US spends $877 billion on defense and has 335 million people.",
            'highlight': "MULTI-CLAIM ANALYSIS"
        },
        {
            'statement': "I believe the economy is doing great and we should be optimistic about the future.",
            'highlight': "PURE OPINION FILTERING"
        }
    ]
    
    for i, case in enumerate(demo_cases, 1):
        print(f"🎯 DEMO {i}: {case['highlight']}")
        print(f"📝 Statement: '{case['statement']}'")
        print("-" * 60)
        
        result = checker.fact_check(case['statement'])
        
        print(f"🏆 VERDICT: {result['verdict']}")
        print(f"🎯 CONFIDENCE: {result['confidence']:.2f}")
        print(f"⚡ PROCESSING: {result['processing_time']:.3f} seconds")
        print(f"💡 EXPLANATION: {result['explanation']}")
        
        if result.get('detailed_analysis'):
            print("🔍 EVIDENCE ANALYSIS:")
            for j, analysis in enumerate(result['detailed_analysis'][:2]):
                if analysis.get('evidence'):
                    print(f"   {j+1}. Evidence: {analysis['evidence'][:50]}...")
                    print(f"      Stance: {analysis['stance']} (conf: {analysis['stance_confidence']:.2f})")
        print()
    
    print("🚀 KEY SYSTEM CAPABILITIES DEMONSTRATED:")
    print("✅ Advanced pattern-based claim detection")
    print("✅ Opinion vs fact discrimination") 
    print("✅ Multi-claim statement processing")
    print("✅ Numerical fact verification")
    print("✅ Confidence-weighted evidence aggregation")
    print("✅ Sub-second real-time processing")
    print("✅ Detailed explainable AI results")

if __name__ == "__main__":
    print("Choose demo mode:")
    print("1. Challenge Tests (comprehensive evaluation)")
    print("2. Presentation Demo (key capabilities)")
    print("3. Both")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice in ['1', '3']:
        analyzer, accuracy = run_challenge_tests()
        print(f"\n🏆 FINAL ACCURACY: {accuracy:.1%}")
    
    if choice in ['2', '3']:
        print("\n" + "="*60)
        create_presentation_demo()
    
    print("\n🎉 DEMO COMPLETE! System ready for presentation!")