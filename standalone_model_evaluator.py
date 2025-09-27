"""
Comprehensive Multi-Model Accuracy Evaluator for Healthcare
Tests ALL available Groq models independently
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from groq import Groq

class HealthcareMultiModelEvaluator:
    """Evaluate accuracy across ALL available Groq models."""
    
    def __init__(self):
        self.api_key = "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"
        self.client = Groq(api_key=self.api_key)
        
        # ALL your available models for testing
        self.models = {
            # Google/Gemma Models
            "gemma2-9b-it": {"category": "Google", "size": "9B", "type": "Instruction"},
            
            # Groq Compound Models
            "groq/compound": {"category": "Groq", "size": "Unknown", "type": "Compound"},
            "groq/compound-mini": {"category": "Groq", "size": "Mini", "type": "Compound"},
            
            # Llama Models
            "llama-3.1-8b-instant": {"category": "Meta", "size": "8B", "type": "Fast"},
            "llama-3.3-70b-versatile": {"category": "Meta", "size": "70B", "type": "Large"},
            
            # Meta Llama 4 Models
            "meta-llama/llama-4-maverick-17b-128": {"category": "Meta", "size": "17B", "type": "Maverick"},
            "meta-llama/llama-4-scout-17b-16e-ins": {"category": "Meta", "size": "17B", "type": "Scout"},
            "meta-llama/llama-guard-4-12b": {"category": "Meta", "size": "12B", "type": "Guard"},
            
            # Moonshot AI Models
            "moonshotai/kimi-k2-instruct": {"category": "Moonshot", "size": "Unknown", "type": "Instruct"},
            "moonshotai/kimi-k2-instruct-0905": {"category": "Moonshot", "size": "Unknown", "type": "Latest"},
            
            # OpenAI Models
            "openai/gpt-oss-120b": {"category": "OpenAI", "size": "120B", "type": "Large"},
            "openai/gpt-oss-20b": {"category": "OpenAI", "size": "20B", "type": "Medium"},
        }
        
        self.results = {}
        
    def get_medical_test_cases(self) -> List[Dict[str, Any]]:
        """Comprehensive medical test cases covering all specialties."""
        return [
            # Emergency Medicine
            {
                "category": "emergency",
                "input": "55-year-old male with crushing chest pain radiating to left arm, diaphoresis, nausea. BP 150/95, HR 110",
                "ground_truth": "myocardial_infarction",
                "keywords": ["heart attack", "MI", "cardiac", "emergency", "STEMI", "ACS", "myocardial"],
                "difficulty": "medium"
            },
            {
                "category": "emergency", 
                "input": "Unconscious patient, pinpoint pupils, slow shallow breathing, track marks visible",
                "ground_truth": "opioid_overdose",
                "keywords": ["overdose", "opioid", "naloxone", "heroin", "respiratory depression"],
                "difficulty": "easy"
            },
            {
                "category": "emergency",
                "input": "Sudden severe headache, neck stiffness, photophobia, fever 102°F",
                "ground_truth": "meningitis",
                "keywords": ["meningitis", "bacterial", "lumbar puncture", "antibiotics", "emergency"],
                "difficulty": "medium"
            },
            
            # Drug Interactions & Pharmacology
            {
                "category": "pharmacology",
                "input": "Patient on warfarin 5mg daily, prescribed ciprofloxacin for UTI. What is the major concern?",
                "ground_truth": "drug_interaction",
                "keywords": ["interaction", "bleeding", "INR", "monitor", "warfarin", "anticoagulation"],
                "difficulty": "medium"
            },
            {
                "category": "pharmacology",
                "input": "Elderly patient taking digoxin, now prescribed furosemide. What monitoring is essential?",
                "ground_truth": "electrolyte_monitoring",
                "keywords": ["potassium", "digoxin toxicity", "electrolytes", "monitor", "hypokalemia"],
                "difficulty": "hard"
            },
            
            # Complex Diagnosis
            {
                "category": "diagnosis",
                "input": "45-year-old female with fatigue, joint pain, malar rash, positive ANA and anti-dsDNA",
                "ground_truth": "systemic_lupus_erythematosus", 
                "keywords": ["SLE", "lupus", "autoimmune", "ANA", "anti-dsDNA", "connective tissue"],
                "difficulty": "hard"
            },
            {
                "category": "diagnosis",
                "input": "Patient with diabetes, blurred vision, excessive thirst, fruity breath, glucose 450 mg/dL",
                "ground_truth": "diabetic_ketoacidosis",
                "keywords": ["DKA", "ketoacidosis", "diabetes", "ketones", "acidosis", "insulin"],
                "difficulty": "medium"
            },
            
            # Infectious Disease
            {
                "category": "infectious_disease",
                "input": "Recent travel to sub-Saharan Africa, fever, chills, headache, low platelets",
                "ground_truth": "malaria",
                "keywords": ["malaria", "plasmodium", "fever", "travel", "tropical", "parasites"],
                "difficulty": "medium"
            },
            {
                "category": "infectious_disease",
                "input": "Hospital patient with prolonged diarrhea after antibiotics, positive C. diff toxin",
                "ground_truth": "c_diff_colitis",
                "keywords": ["C diff", "colitis", "antibiotic", "diarrhea", "vancomycin", "metronidazole"],
                "difficulty": "easy"
            },
            
            # Pediatrics
            {
                "category": "pediatrics",
                "input": "2-year-old with barking cough, inspiratory stridor, fever 101°F, recent URI",
                "ground_truth": "croup",
                "keywords": ["croup", "stridor", "barking cough", "viral", "laryngotracheitis", "steroid"],
                "difficulty": "easy"
            }
        ]
        
    async def test_model_on_case(self, model_name: str, test_case: Dict) -> Dict[str, Any]:
        """Test a single model on one test case."""
        try:
            # Create focused medical prompt
            prompt = f"""You are an expert physician. Analyze this clinical scenario and provide a concise medical assessment.

Clinical Scenario: {test_case['input']}

Provide your response in this exact format:
Primary Diagnosis: [your primary diagnosis/assessment]
Confidence: [percentage 0-100%]
Reasoning: [brief clinical reasoning]
"""
            
            # Get model response
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_completion_tokens=200
            )
            
            prediction = response.choices[0].message.content
            
            # Evaluate accuracy
            is_correct = self._evaluate_medical_prediction(
                prediction, 
                test_case["ground_truth"], 
                test_case["keywords"]
            )
            
            return {
                "model": model_name,
                "case_category": test_case["category"],
                "difficulty": test_case["difficulty"],
                "prediction": prediction,
                "is_correct": is_correct,
                "ground_truth": test_case["ground_truth"],
                "success": True
            }
            
        except Exception as e:
            return {
                "model": model_name,
                "case_category": test_case["category"],
                "difficulty": test_case["difficulty"], 
                "prediction": f"ERROR: {str(e)}",
                "is_correct": False,
                "ground_truth": test_case["ground_truth"],
                "success": False,
                "error": str(e)
            }
    
    def _evaluate_medical_prediction(self, prediction: str, ground_truth: str, keywords: List[str]) -> bool:
        """Evaluate if medical prediction is accurate."""
        prediction_lower = prediction.lower()
        ground_truth_lower = ground_truth.lower().replace("_", " ")
        
        # Direct match check
        if ground_truth_lower in prediction_lower:
            return True
        
        # Keyword matching - need at least 2 relevant keywords
        keyword_matches = sum(1 for keyword in keywords if keyword.lower() in prediction_lower)
        return keyword_matches >= 2
    
    async def evaluate_all_models(self) -> Dict[str, Any]:
        """Test every model on every test case."""
        print("🚀 COMPREHENSIVE MULTI-MODEL HEALTHCARE EVALUATION")
        print("=" * 80)
        print(f"📊 Testing {len(self.models)} models on {len(self.get_medical_test_cases())} medical cases")
        print()
        
        test_cases = self.get_medical_test_cases()
        all_results = {}
        
        for i, model_name in enumerate(self.models, 1):
            print(f"🔬 Testing Model {i}/{len(self.models)}: {model_name}")
            
            model_results = {
                "model_info": self.models[model_name],
                "total_cases": len(test_cases),
                "correct": 0,
                "accuracy": 0.0,
                "category_performance": {},
                "difficulty_performance": {},
                "case_results": []
            }
            
            for j, test_case in enumerate(test_cases, 1):
                print(f"   Case {j}/{len(test_cases)}: {test_case['category']}", end=" ")
                
                result = await self.test_model_on_case(model_name, test_case)
                model_results["case_results"].append(result)
                
                if result["is_correct"]:
                    model_results["correct"] += 1
                    print("✅")
                else:
                    print("❌")
                
                # Track performance by category
                category = test_case["category"]
                if category not in model_results["category_performance"]:
                    model_results["category_performance"][category] = {"correct": 0, "total": 0}
                model_results["category_performance"][category]["total"] += 1
                if result["is_correct"]:
                    model_results["category_performance"][category]["correct"] += 1
                
                # Track performance by difficulty
                difficulty = test_case["difficulty"]
                if difficulty not in model_results["difficulty_performance"]:
                    model_results["difficulty_performance"][difficulty] = {"correct": 0, "total": 0}
                model_results["difficulty_performance"][difficulty]["total"] += 1
                if result["is_correct"]:
                    model_results["difficulty_performance"][difficulty]["correct"] += 1
            
            # Calculate final accuracy
            model_results["accuracy"] = model_results["correct"] / model_results["total_cases"]
            
            # Calculate category accuracies
            for category in model_results["category_performance"]:
                perf = model_results["category_performance"][category]
                perf["accuracy"] = perf["correct"] / perf["total"] if perf["total"] > 0 else 0
            
            # Calculate difficulty accuracies  
            for difficulty in model_results["difficulty_performance"]:
                perf = model_results["difficulty_performance"][difficulty]
                perf["accuracy"] = perf["correct"] / perf["total"] if perf["total"] > 0 else 0
            
            all_results[model_name] = model_results
            
            print(f"   🎯 Model Accuracy: {model_results['accuracy']:.1%} ({model_results['correct']}/{model_results['total_cases']})")
            print()
        
        self.results = all_results
        return all_results
    
    def generate_comprehensive_report(self) -> str:
        """Generate detailed accuracy report for all models."""
        if not self.results:
            return "No evaluation results available."
        
        # Sort models by accuracy
        sorted_models = sorted(self.results.items(), 
                             key=lambda x: x[1]["accuracy"], reverse=True)
        
        report = f"""
🏥 COMPREHENSIVE HEALTHCARE AI MODEL ACCURACY REPORT
{'=' * 100}

📊 EVALUATION OVERVIEW
• Models Tested: {len(self.results)}
• Test Cases: {len(self.get_medical_test_cases())}
• Medical Specialties: Emergency, Pharmacology, Diagnosis, Infectious Disease, Pediatrics
• Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🏆 MODEL PERFORMANCE RANKINGS
{'=' * 50}
"""
        
        for rank, (model_name, results) in enumerate(sorted_models, 1):
            accuracy = results["accuracy"]
            category = results["model_info"]["category"]
            size = results["model_info"]["size"]
            model_type = results["model_info"]["type"]
            
            # Performance tier
            if accuracy >= 0.9:
                tier = "🥇 EXCELLENT"
            elif accuracy >= 0.8:
                tier = "🥈 VERY GOOD"  
            elif accuracy >= 0.7:
                tier = "🥉 GOOD"
            elif accuracy >= 0.5:
                tier = "📊 FAIR"
            else:
                tier = "⚠️ NEEDS IMPROVEMENT"
            
            report += f"""
{tier} | Rank #{rank}: {model_name}
   📈 Accuracy: {accuracy:.2%} ({results['correct']}/{results['total_cases']})
   🏷️ Details: {category} | {size} | {model_type}
"""
            
            # Show category breakdown for top 3
            if rank <= 3:
                report += "   📋 Category Performance:\n"
                for cat, perf in results["category_performance"].items():
                    cat_acc = perf["accuracy"]
                    report += f"      • {cat.replace('_', ' ').title()}: {cat_acc:.1%}\n"
        
        # Overall insights
        report += f"""

💡 KEY INSIGHTS
{'=' * 30}
🏆 Best Overall Model: {sorted_models[0][0]} ({sorted_models[0][1]['accuracy']:.1%})
🎯 Average Accuracy: {np.mean([r['accuracy'] for r in self.results.values()]):.1%}
📊 Models Above 70%: {sum(1 for r in self.results.values() if r['accuracy'] >= 0.7)}
"""
        
        # Category analysis
        category_stats = {}
        for results in self.results.values():
            for cat, perf in results["category_performance"].items():
                if cat not in category_stats:
                    category_stats[cat] = []
                category_stats[cat].append(perf["accuracy"])
        
        report += f"""

📋 PERFORMANCE BY MEDICAL SPECIALTY
{'=' * 40}
"""
        
        for category, accuracies in category_stats.items():
            avg_acc = np.mean(accuracies)
            best_acc = np.max(accuracies)
            
            # Find best model for this category
            best_model = None
            for model_name, results in self.results.items():
                if category in results["category_performance"]:
                    if results["category_performance"][category]["accuracy"] == best_acc:
                        best_model = model_name
                        break
            
            report += f"""
• {category.replace('_', ' ').title()}:
  Average: {avg_acc:.1%} | Best: {best_acc:.1%} 
  Top Performer: {best_model}
"""
        
        return report
    
    def create_visualization(self):
        """Create comprehensive visualization of results."""
        if not self.results:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        
        # 1. Overall Model Accuracy Ranking
        models = []
        accuracies = []
        colors = []
        
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1]["accuracy"], reverse=True)
        
        for model_name, results in sorted_results:
            models.append(model_name.split('/')[-1][:15])  # Truncate long names
            accuracies.append(results["accuracy"])
            
            # Color by performance tier
            acc = results["accuracy"]
            if acc >= 0.8:
                colors.append('#2E8B57')  # Green
            elif acc >= 0.6:
                colors.append('#4169E1')  # Blue  
            elif acc >= 0.4:
                colors.append('#FF8C00')  # Orange
            else:
                colors.append('#DC143C')  # Red
        
        bars = ax1.barh(models, accuracies, color=colors, alpha=0.8)
        ax1.set_title('Healthcare AI Model Accuracy Rankings', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Accuracy Score')
        ax1.set_xlim(0, 1)
        
        # Add percentage labels
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{acc:.1%}', va='center', fontweight='bold', fontsize=10)
        
        # 2. Performance by Model Category
        category_performance = {}
        for model_name, results in self.results.items():
            category = results["model_info"]["category"]
            if category not in category_performance:
                category_performance[category] = []
            category_performance[category].append(results["accuracy"])
        
        categories = list(category_performance.keys())
        avg_accuracies = [np.mean(category_performance[cat]) for cat in categories]
        
        ax2.bar(categories, avg_accuracies, color='skyblue', alpha=0.7)
        ax2.set_title('Average Accuracy by Model Provider', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Accuracy')
        ax2.tick_params(axis='x', rotation=45)
        
        for i, acc in enumerate(avg_accuracies):
            ax2.text(i, acc + 0.01, f'{acc:.1%}', ha='center', fontweight='bold')
        
        # 3. Medical Specialty Performance Heatmap
        specialties = list(set())
        for results in self.results.values():
            specialties.extend(results["category_performance"].keys())
        specialties = sorted(list(set(specialties)))
        
        # Fix: Correctly access model names from tuples
        model_names = [model_tuple[0].split('/')[-1][:10] for model_tuple in sorted_results[:8]]  # Top 8 models

        heatmap_data = []
        for model_name, results in sorted_results[:8]:
            row = []
            for specialty in specialties:
                if specialty in results["category_performance"]:
                    row.append(results["category_performance"][specialty]["accuracy"])
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        if heatmap_data and len(heatmap_data) > 0:
            im = ax3.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax3.set_title('Top Models: Performance by Medical Specialty', fontsize=14, fontweight='bold')
            ax3.set_xticks(range(len(specialties)))
            ax3.set_xticklabels([s.replace('_', ' ').title()[:10] for s in specialties], rotation=45, ha='right')
            ax3.set_yticks(range(len(model_names)))
            ax3.set_yticklabels(model_names)

            # Add colorbar
            plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

        # 4. Distribution of Accuracies
        all_accuracies = [results["accuracy"] for results in self.results.values()]
        
        ax4.hist(all_accuracies, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.set_title('Distribution of Model Accuracies', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Accuracy Score')
        ax4.set_ylabel('Number of Models')
        ax4.axvline(np.mean(all_accuracies), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_accuracies):.1%}')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save visualization
        output_path = Path("evaluation/comprehensive_model_comparison.png")
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        print(f"📊 Comprehensive visualization saved to: {output_path}")
        plt.show()
        
        return output_path
    
    def export_results(self):
        """Export detailed results to JSON."""
        output_data = {
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_models": len(self.results),
                "total_test_cases": len(self.get_medical_test_cases()),
                "specialties_tested": ["emergency", "pharmacology", "diagnosis", "infectious_disease", "pediatrics"]
            },
            "model_results": self.results
        }
        
        output_file = Path(f"evaluation/comprehensive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"📁 Complete results exported to: {output_file}")
        return str(output_file)

async def main():
    """Run comprehensive evaluation of all models."""
    evaluator = HealthcareMultiModelEvaluator()
    
    # Run evaluation
    await evaluator.evaluate_all_models()
    
    # Generate and display report
    report = evaluator.generate_comprehensive_report()
    print(report)
    
    # Create visualizations
    evaluator.create_visualization()
    
    # Export results
    evaluator.export_results()
    
    print("\n🎉 EVALUATION COMPLETE!")
    print("All models tested for healthcare accuracy across multiple medical specialties.")

if __name__ == "__main__":
    asyncio.run(main())
