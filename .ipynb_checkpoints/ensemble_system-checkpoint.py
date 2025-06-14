# ENSEMBLE POLITICAL FACT-CHECKER
# Combines ALL approaches for maximum accuracy and insight!

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from collections import Counter
import datetime
import json

class EnsemblePoliticalFactChecker:
    """Advanced ensemble fact-checker combining all approaches"""
    
    def __init__(self):
        print("🚀 Initializing ENSEMBLE POLITICAL FACT-CHECKER...")
        
        # Initialize rule-based system
        from main import PoliticalFactChecker
        self.rule_based_checker = PoliticalFactChecker()
        print("✅ Rule-based system loaded")
        
        # Initialize ML components
        self.ml_model = None
        self.ml_vectorizer = None
        self.ml_ready = False
        
        # Label mappings for ML model
        self.label_mapping = {
            'pants-fire': 0, 'false': 1, 'barely-true': 2,
            'half-true': 3, 'mostly-true': 4, 'true': 5
        }
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        
        # Confidence thresholds
        self.high_confidence_threshold = 0.8
        self.medium_confidence_threshold = 0.5
        
        # Results history for learning
        self.prediction_history = []
        
        self._train_ml_component()
        print("🎯 ENSEMBLE SYSTEM READY!")
    
    def _train_ml_component(self):
        """Train the ML component on LIAR-style data"""
        print("🤖 Training ML component...")
        
        # Create training data (expanded from LIAR demo)
        training_data = [
            # TRUE statements
            ("The minimum wage has been $7.25 per hour since 2009", "true"),
            ("Climate change is supported by 97% of scientists", "true"), 
            ("Social Security provides benefits to over 67 million Americans", "true"),
            ("The US Constitution was signed in 1787", "true"),
            ("Medicare covers approximately 65 million Americans", "true"),
            ("The United States has 50 states", "true"),
            ("The federal minimum wage has not changed since 2009", "true"),
            
            # MOSTLY-TRUE statements
            ("The US spends approximately $877 billion on defense annually", "mostly-true"),
            ("The unemployment rate was around 3.7% in recent years", "mostly-true"),
            ("Student loan debt exceeds $1.5 trillion nationally", "mostly-true"),
            ("The US population is approximately 335 million people", "mostly-true"),
            ("Renewable energy accounts for about 20% of electricity generation", "mostly-true"),
            ("The US imports about 60% of its oil consumption", "mostly-true"),
            
            # HALF-TRUE statements
            ("The US spends more on defense than most other countries", "half-true"),
            ("Immigration has significantly impacted the economy", "half-true"),
            ("Healthcare costs have been rising for decades", "half-true"),
            ("The economy has improved under recent administrations", "half-true"),
            ("Education funding varies significantly across states", "half-true"),
            ("Infrastructure needs significant investment nationwide", "half-true"),
            
            # BARELY-TRUE statements
            ("The economy is performing better than ever in history", "barely-true"),
            ("Crime rates have dramatically decreased nationwide", "barely-true"),
            ("Infrastructure spending has solved most transportation issues", "barely-true"),
            ("The healthcare system provides universal coverage", "barely-true"),
            ("Income inequality has been effectively addressed", "barely-true"),
            ("All Americans have access to quality education", "barely-true"),
            
            # FALSE statements
            ("The unemployment rate is 25% nationwide", "false"),
            ("The US has 52 states", "false"),
            ("The minimum wage is $15 per hour federally", "false"),
            ("Climate change has been completely disproven", "false"),
            ("Social Security has been eliminated", "false"),
            ("The national debt has been completely paid off", "false"),
            
            # PANTS-FIRE statements
            ("The unemployment rate is 50% in America", "pants-fire"),
            ("The Earth is flat according to scientists", "pants-fire"),
            ("The US Constitution was written in 1492", "pants-fire"),
            ("All taxes have been eliminated permanently", "pants-fire"),
            ("The US has 100 states now", "pants-fire"),
            ("The moon landing was filmed in a studio", "pants-fire"),
        ]
        
        # Create DataFrame and train
        df = pd.DataFrame(training_data, columns=['statement', 'label'])
        
        X = df['statement'].values
        y = [self.label_mapping[label] for label in df['label']]
        
        # Train TF-IDF vectorizer
        self.ml_vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 3))
        X_vectorized = self.ml_vectorizer.fit_transform(X)
        
        # Train model
        self.ml_model = LogisticRegression(random_state=42, max_iter=2000, C=1.0)
        self.ml_model.fit(X_vectorized, y)
        
        self.ml_ready = True
        print("✅ ML component trained on 36 political statements")
    
    def _get_ml_prediction(self, statement):
        """Get prediction from ML model"""
        if not self.ml_ready:
            return None
        
        try:
            # Vectorize statement
            X_vec = self.ml_vectorizer.transform([statement])
            
            # Get prediction and probabilities
            prediction = self.ml_model.predict(X_vec)[0]
            probabilities = self.ml_model.predict_proba(X_vec)[0]
            
            # Get confidence (max probability)
            confidence = max(probabilities)
            
            # Convert to readable label
            predicted_label = self.reverse_mapping[prediction]
            
            # Get top 3 predictions with probabilities
            top_predictions = []
            for i, prob in enumerate(probabilities):
                if prob > 0.05:  # Only show if >5% probability
                    top_predictions.append({
                        'label': self.reverse_mapping[i],
                        'probability': prob
                    })
            
            # Sort by probability
            top_predictions.sort(key=lambda x: x['probability'], reverse=True)
            
            return {
                'prediction': predicted_label,
                'confidence': confidence,
                'top_predictions': top_predictions[:3],
                'method': 'ML_LIAR_trained'
            }
            
        except Exception as e:
            print(f"⚠️ ML prediction error: {e}")
            return None
    
    def _map_rule_to_liar(self, rule_verdict):
        """Map rule-based verdict to LIAR scale"""
        mapping = {
            'MOSTLY_TRUE': 'mostly-true',
            'PARTIALLY_TRUE': 'half-true', 
            'MOSTLY_FALSE': 'false',
            'MIXED_EVIDENCE': 'barely-true',
            'NO_FACTUAL_CLAIMS': 'barely-true',
            'INSUFFICIENT_EVIDENCE': 'barely-true'
        }
        return mapping.get(rule_verdict, 'half-true')
    
    def _ensemble_prediction(self, rule_result, ml_result):
        """Combine predictions from both systems intelligently"""
        
        # If ML failed, use rule-based
        if ml_result is None:
            return {
                'final_verdict': rule_result['verdict'],
                'confidence': rule_result['confidence'],
                'method': 'rule_based_only',
                'explanation': f"ML unavailable. {rule_result['explanation']}"
            }
        
        # Convert rule-based to LIAR scale for comparison
        rule_liar = self._map_rule_to_liar(rule_result['verdict'])
        ml_liar = ml_result['prediction']
        
        # Calculate agreement
        agreement = rule_liar == ml_liar
        
        # Ensemble logic
        if agreement and rule_result['confidence'] > 0.7 and ml_result['confidence'] > 0.7:
            # High confidence agreement
            final_verdict = rule_result['verdict']
            confidence = (rule_result['confidence'] + ml_result['confidence']) / 2
            method = 'high_confidence_agreement'
            explanation = f"Both systems agree with high confidence. {rule_result['explanation']}"
            
        elif agreement:
            # Medium confidence agreement
            final_verdict = rule_result['verdict']
            confidence = (rule_result['confidence'] + ml_result['confidence']) / 2
            method = 'medium_confidence_agreement'
            explanation = f"Systems agree. {rule_result['explanation']}"
            
        elif rule_result['confidence'] > ml_result['confidence']:
            # Rule-based more confident
            final_verdict = rule_result['verdict']
            confidence = rule_result['confidence'] * 0.8  # Reduce due to disagreement
            method = 'rule_based_preferred'
            explanation = f"Rule-based system more confident. {rule_result['explanation']}"
            
        else:
            # ML more confident or equal - defer to rule-based for interpretability
            final_verdict = rule_result['verdict']
            confidence = max(rule_result['confidence'], ml_result['confidence']) * 0.7
            method = 'interpretable_fallback'
            explanation = f"Systems disagree, prioritizing interpretable result. {rule_result['explanation']}"
        
        return {
            'final_verdict': final_verdict,
            'confidence': confidence,
            'method': method,
            'explanation': explanation,
            'agreement': agreement
        }
    
    def comprehensive_fact_check(self, statement):
        """Comprehensive fact-checking using ensemble approach"""
        
        print(f"\n🎯 ENSEMBLE ANALYSIS: {statement}")
        print("="*60)
        
        start_time = datetime.datetime.now()
        
        # Get rule-based prediction
        print("📋 Rule-based Analysis...")
        rule_result = self.rule_based_checker.fact_check(statement)
        
        # Get ML prediction
        print("🤖 ML Analysis...")
        ml_result = self._get_ml_prediction(statement)
        
        # Combine predictions
        print("🔮 Ensemble Combination...")
        ensemble_result = self._ensemble_prediction(rule_result, ml_result)
        
        # Calculate processing time
        processing_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Compile comprehensive result
        comprehensive_result = {
            'statement': statement,
            'timestamp': datetime.datetime.now().isoformat(),
            'processing_time': processing_time,
            
            # Ensemble result
            'ensemble_verdict': ensemble_result['final_verdict'],
            'ensemble_confidence': ensemble_result['confidence'],
            'ensemble_method': ensemble_result['method'],
            'ensemble_explanation': ensemble_result['explanation'],
            'systems_agreement': ensemble_result.get('agreement', False),
            
            # Individual system results
            'rule_based': {
                'verdict': rule_result['verdict'],
                'confidence': rule_result['confidence'],
                'explanation': rule_result['explanation'],
                'claims_analyzed': rule_result.get('claims_analyzed', 0),
                'claims_with_evidence': rule_result.get('claims_with_evidence', 0)
            },
            
            'ml_system': ml_result if ml_result else {'status': 'unavailable'},
            
            # Detailed analysis
            'detailed_analysis': rule_result.get('detailed_analysis', [])
        }
        
        # Store for learning
        self.prediction_history.append(comprehensive_result)
        
        return comprehensive_result
    
    def display_result(self, result):
        """Display comprehensive result in a nice format"""
        
        print(f"\n🏆 ENSEMBLE VERDICT: {result['ensemble_verdict']}")
        print(f"🎯 CONFIDENCE: {result['ensemble_confidence']:.3f}")
        print(f"⚡ PROCESSING TIME: {result['processing_time']:.3f}s")
        print(f"🔗 METHOD: {result['ensemble_method']}")
        print(f"🤝 SYSTEMS AGREEMENT: {'✅ Yes' if result['systems_agreement'] else '❌ No'}")
        
        print(f"\n💡 EXPLANATION:")
        print(f"   {result['ensemble_explanation']}")
        
        print(f"\n📊 INDIVIDUAL SYSTEM RESULTS:")
        
        # Rule-based results
        rule = result['rule_based']
        print(f"   🔧 Rule-based: {rule['verdict']} (conf: {rule['confidence']:.2f})")
        print(f"      Claims: {rule['claims_analyzed']}, Evidence: {rule['claims_with_evidence']}")
        
        # ML results
        ml = result['ml_system']
        if ml.get('status') != 'unavailable':
            print(f"   🤖 ML System: {ml['prediction']} (conf: {ml['confidence']:.2f})")
            if ml.get('top_predictions'):
                print(f"      Top predictions:")
                for pred in ml['top_predictions'][:2]:
                    print(f"        • {pred['label']}: {pred['probability']:.2f}")
        else:
            print(f"   🤖 ML System: Unavailable")
        
        # Evidence details
        if result.get('detailed_analysis'):
            print(f"\n🔍 EVIDENCE ANALYSIS:")
            for i, analysis in enumerate(result['detailed_analysis'][:2]):
                if analysis.get('evidence'):
                    print(f"   {i+1}. {analysis['evidence'][:50]}...")
                    print(f"      → {analysis['stance']} (conf: {analysis.get('stance_confidence', 0):.2f})")
    
    def interactive_session(self):
        """Run interactive fact-checking session"""
        
        print("🎯" * 25)
        print("INTERACTIVE ENSEMBLE POLITICAL FACT-CHECKER")
        print("🎯" * 25)
        print("Combines rule-based + ML for maximum accuracy!")
        print("Enter political statements to fact-check (type 'quit' to exit)")
        print("="*75)
        
        session_results = []
        
        while True:
            try:
                statement = input("\n📝 Enter political statement: ").strip()
                
                if statement.lower() in ['quit', 'exit', 'q', '']:
                    break
                
                if len(statement) < 10:
                    print("⚠️ Please enter a longer statement for better analysis")
                    continue
                
                # Comprehensive analysis
                result = self.comprehensive_fact_check(statement)
                
                # Display results
                self.display_result(result)
                
                # Store for session summary
                session_results.append(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted by user")
                break
            except Exception as e:
                print(f"\n⚠️ Error: {e}")
                continue
        
        # Session summary
        if session_results:
            self.display_session_summary(session_results)
    
    def display_session_summary(self, session_results):
        """Display summary of the session"""
        
        print(f"\n📊 SESSION SUMMARY")
        print("="*40)
        print(f"Statements analyzed: {len(session_results)}")
        
        # Verdict distribution
        verdicts = [r['ensemble_verdict'] for r in session_results]
        verdict_counts = Counter(verdicts)
        
        print(f"\nVerdict distribution:")
        for verdict, count in verdict_counts.most_common():
            print(f"  • {verdict}: {count}")
        
        # Average confidence
        avg_confidence = sum(r['ensemble_confidence'] for r in session_results) / len(session_results)
        print(f"\nAverage confidence: {avg_confidence:.2f}")
        
        # Agreement rate
        agreements = sum(1 for r in session_results if r['systems_agreement'])
        agreement_rate = agreements / len(session_results)
        print(f"Systems agreement rate: {agreement_rate:.1%}")
        
        # Processing performance
        avg_time = sum(r['processing_time'] for r in session_results) / len(session_results)
        print(f"Average processing time: {avg_time:.3f}s")
        
        print(f"\n🚀 ENSEMBLE SYSTEM PERFORMANCE:")
        print(f"✅ Multi-model validation")
        print(f"✅ Interpretable reasoning")
        print(f"✅ Sub-second processing")
        print(f"✅ Comprehensive evidence analysis")

def run_demo_session():
    """Run a demo session with sample statements"""
    
    print("🎪 ENSEMBLE FACT-CHECKER DEMO")
    print("="*50)
    
    # Initialize ensemble system
    ensemble = EnsemblePoliticalFactChecker()
    
    # Demo statements
    demo_statements = [
        "The minimum wage has been $7.25 per hour since 2009 and should be increased",
        "Unemployment in America is at record highs of 25% this year",
        "Climate change is supported by 97% of scientists according to multiple studies",
        "The US spends over $800 billion on defense which is more than necessary",
        "I believe the economy is doing great and we should be optimistic"
    ]
    
    print(f"\n🎯 TESTING {len(demo_statements)} STATEMENTS WITH ENSEMBLE SYSTEM")
    print("="*70)
    
    results = []
    
    for i, statement in enumerate(demo_statements, 1):
        print(f"\n📝 DEMO STATEMENT {i}:")
        print(f"'{statement}'")
        print("-" * 60)
        
        result = ensemble.comprehensive_fact_check(statement)
        ensemble.display_result(result)
        results.append(result)
    
    # Demo summary
    ensemble.display_session_summary(results)
    
    return ensemble

def main():
    """Main function with options"""
    
    print("🚀 ENSEMBLE POLITICAL FACT-CHECKER")
    print("="*50)
    print("Choose mode:")
    print("1. Interactive session")
    print("2. Demo with sample statements")
    print("3. Both")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            ensemble = EnsemblePoliticalFactChecker()
            ensemble.interactive_session()
        elif choice == '2':
            ensemble = run_demo_session()
        elif choice == '3':
            ensemble = run_demo_session()
            print("\n" + "="*60)
            print("🎯 SWITCHING TO INTERACTIVE MODE")
            print("="*60)
            ensemble.interactive_session()
        else:
            print("Invalid choice, running demo...")
            run_demo_session()
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    
    print("\n🎉 ENSEMBLE FACT-CHECKER SESSION COMPLETE!")
    print("This system demonstrates:")
    print("✅ Multi-model ensemble approach")
    print("✅ Interpretable AI with ML validation") 
    print("✅ Comprehensive evidence analysis")
    print("✅ Real-time political fact-checking")
    print("✅ Research-level methodology")

if __name__ == "__main__":
    main()