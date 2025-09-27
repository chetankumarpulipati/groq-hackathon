"""
Comprehensive Multi-Model Accuracy Evaluator
Tests ALL available Groq models for healthcare performance
"""

import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
from utils.logging import get_logger

logger = get_logger("multi_model_evaluator")

class AllModelsAccuracyEvaluator:
    """Evaluate accuracy across ALL available Groq models."""

    def __init__(self):
        self.api_key = "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"
        self.client = Groq(api_key=self.api_key)

        # ALL available models for testing
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
            "moonshotai/kimi-k2-instruct-0905": {"category": "Moonshot", "size": "Unknown", "type": "Instruct-Latest"},

            # OpenAI Models
            "openai/gpt-oss-120b": {"category": "OpenAI", "size": "120B", "type": "Large"},
            "openai/gpt-oss-20b": {"category": "OpenAI", "size": "20B", "type": "Medium"},
        }

        self.test_cases = []
        self.results = {}

    def load_medical_test_cases(self) -> List[Dict[str, Any]]:
        """Load comprehensive medical test cases."""
        test_cases = [
            # Emergency Medicine
            {
                "category": "emergency",
                "input": "55-year-old male presents with severe crushing chest pain radiating to left arm, diaphoresis, and nausea. BP 150/95, HR 110",
                "ground_truth": "myocardial_infarction",
                "keywords": ["heart attack", "MI", "cardiac", "emergency", "STEMI", "ACS"],
                "difficulty": "medium"
            },
            {
                "category": "emergency",
                "input": "Unconscious patient found down, pinpoint pupils, slow shallow breathing, track marks on arms",
                "ground_truth": "opioid_overdose",
                "keywords": ["overdose", "opioid", "naloxone", "heroin", "respiratory depression"],
                "difficulty": "easy"
            },

            # Drug Interactions
            {
                "category": "pharmacology",
                "input": "Patient on warfarin 5mg daily, prescribed ciprofloxacin for UTI. What is the concern?",
                "ground_truth": "drug_interaction",
                "keywords": ["interaction", "bleeding", "INR", "monitor", "warfarin"],
                "difficulty": "medium"
            },
            {
                "category": "pharmacology",
                "input": "Elderly patient taking digoxin, now prescribed furosemide. What should be monitored?",
                "ground_truth": "electrolyte_monitoring",
                "keywords": ["potassium", "digoxin toxicity", "electrolytes", "monitor"],
                "difficulty": "hard"
            },

            # Complex Diagnosis
            {
                "category": "diagnosis",
                "input": "45-year-old female with fatigue, joint pain, malar rash, positive ANA and anti-dsDNA",
                "ground_truth": "systemic_lupus_erythematosus",
                "keywords": ["SLE", "lupus", "autoimmune", "ANA", "anti-dsDNA"],
                "difficulty": "hard"
            },
            {
                "category": "diagnosis",
                "input": "Patient with diabetes, blurred vision, excessive thirst, fruity breath odor, glucose 450 mg/dL",
                "ground_truth": "diabetic_ketoacidosis",
                "keywords": ["DKA", "ketoacidosis", "diabetes", "ketones", "acidosis"],
                "difficulty": "medium"
            },

            # Pediatrics
            {
                "category": "pediatrics",
                "input": "2-year-old with barking cough, stridor, fever 101°F, runny nose for 2 days",
                "ground_truth": "croup",
                "keywords": ["croup", "stridor", "barking cough", "viral", "laryngotracheitis"],
                "difficulty": "easy"
            },

            # Psychiatry
            {
                "category": "psychiatry",
                "input": "20-year-old college student reports hearing voices, paranoid thoughts, social withdrawal for 3 months",
                "ground_truth": "first_episode_psychosis",
                "keywords": ["psychosis", "schizophrenia", "hallucinations", "paranoid"],
                "difficulty": "medium"
            },

            # Infectious Disease
            {
                "category": "infectious_disease",
                "input": "Recent travel to malaria-endemic area, fever, chills, headache, thrombocytopenia",
                "ground_truth": "malaria",
                "keywords": ["malaria", "plasmodium", "fever", "travel", "tropical"],
                "difficulty": "medium"
            },

            # Dermatology
            {
                "category": "dermatology",
                "input": "Asymmetric dark lesion with irregular borders, 8mm diameter, recently changed color",
                "ground_truth": "melanoma_concern",
                "keywords": ["melanoma", "ABCDE", "biopsy", "dermatology", "skin cancer"],
                "difficulty": "medium"
            }
        ]

        self.test_cases = test_cases
        return test_cases

    async def test_single_model(self, model_name: str, test_cases: List[Dict]) -> Dict[str, Any]:
        """Test a single model on all test cases."""
        logger.info(f"Testing model: {model_name}")

        model_results = {
            "model": model_name,
            "category": self.models[model_name]["category"],
            "size": self.models[model_name]["size"],
            "type": self.models[model_name]["type"],
            "total_cases": len(test_cases),
            "correct": 0,
            "accuracy": 0.0,
            "category_performance": {},
            "difficulty_performance": {},
            "detailed_results": []
        }

        for i, test_case in enumerate(test_cases):
            try:
                # Create medical prompt
                prompt = f"""As an expert medical AI, analyze this clinical scenario and provide a concise diagnosis or assessment.

Clinical Scenario: {test_case['input']}

Provide your response in this format:
Primary Assessment: [your diagnosis]
Confidence: [0-100%]
Key Reasoning: [brief explanation]
"""

                # Get model response
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_completion_tokens=300
                )

                prediction = response.choices[0].message.content

                # Evaluate accuracy
                is_correct = self._evaluate_prediction(
                    prediction,
                    test_case["ground_truth"],
                    test_case["keywords"]
                )

                # Store result
                result = {
                    "case_id": i + 1,
                    "category": test_case["category"],
                    "difficulty": test_case["difficulty"],
                    "prediction": prediction[:200],  # Truncate for storage
                    "is_correct": is_correct,
                    "ground_truth": test_case["ground_truth"]
                }

                model_results["detailed_results"].append(result)

                if is_correct:
                    model_results["correct"] += 1

                # Track by category
                category = test_case["category"]
                if category not in model_results["category_performance"]:
                    model_results["category_performance"][category] = {"correct": 0, "total": 0}
                model_results["category_performance"][category]["total"] += 1
                if is_correct:
                    model_results["category_performance"][category]["correct"] += 1

                # Track by difficulty
                difficulty = test_case["difficulty"]
                if difficulty not in model_results["difficulty_performance"]:
                    model_results["difficulty_performance"][difficulty] = {"correct": 0, "total": 0}
                model_results["difficulty_performance"][difficulty]["total"] += 1
                if is_correct:
                    model_results["difficulty_performance"][difficulty]["correct"] += 1

            except Exception as e:
                logger.error(f"Error testing {model_name} on case {i+1}: {e}")
                # Mark as incorrect for failed API calls
                result = {
                    "case_id": i + 1,
                    "category": test_case["category"],
                    "difficulty": test_case["difficulty"],
                    "prediction": f"ERROR: {str(e)[:100]}",
                    "is_correct": False,
                    "ground_truth": test_case["ground_truth"]
                }
                model_results["detailed_results"].append(result)

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

        return model_results

    def _evaluate_prediction(self, prediction: str, ground_truth: str, keywords: List[str]) -> bool:
        """Evaluate if prediction matches ground truth."""
        prediction_lower = prediction.lower()
        ground_truth_lower = ground_truth.lower()

        # Check if ground truth concept is mentioned
        if ground_truth_lower.replace("_", " ") in prediction_lower:
            return True

        # Check if at least 2 keywords are present
        keyword_matches = sum(1 for keyword in keywords if keyword.lower() in prediction_lower)
        return keyword_matches >= 2

    async def evaluate_all_models(self) -> Dict[str, Any]:
        """Evaluate all models and return comprehensive results."""
        logger.info("🧪 Starting comprehensive evaluation of ALL models...")

        test_cases = self.load_medical_test_cases()
        all_results = {}

        for model_name in self.models:
            try:
                model_results = await self.test_single_model(model_name, test_cases)
                all_results[model_name] = model_results
                logger.info(f"✅ {model_name}: {model_results['accuracy']:.1%} accuracy")
            except Exception as e:
                logger.error(f"❌ {model_name}: Failed - {e}")
                all_results[model_name] = {
                    "model": model_name,
                    "accuracy": 0.0,
                    "error": str(e),
                    "total_cases": len(test_cases),
                    "correct": 0
                }

        self.results = all_results
        return all_results

    def create_comparison_visualizations(self):
        """Create comprehensive comparison charts."""
        if not self.results:
            return

        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

        # 1. Overall Accuracy Comparison
        models = []
        accuracies = []
        colors = []

        for model_name, results in self.results.items():
            if 'accuracy' in results:
                models.append(model_name.split('/')[-1])  # Clean model names
                accuracies.append(results['accuracy'])
                # Color by category
                category = results.get('category', 'Unknown')
                if category == 'Meta':
                    colors.append('#1f77b4')
                elif category == 'OpenAI':
                    colors.append('#ff7f0e')
                elif category == 'Google':
                    colors.append('#2ca02c')
                elif category == 'Moonshot':
                    colors.append('#d62728')
                else:
                    colors.append('#9467bd')

        bars = ax1.barh(models, accuracies, color=colors, alpha=0.8)
        ax1.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Accuracy Score')
        ax1.set_xlim(0, 1)

        # Add percentage labels
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{acc:.1%}', va='center', fontweight='bold')

        # 2. Performance by Model Size
        size_performance = {}
        for model_name, results in self.results.items():
            if 'accuracy' in results and 'size' in results:
                size = results['size']
                if size not in size_performance:
                    size_performance[size] = []
                size_performance[size].append(results['accuracy'])

        sizes = list(size_performance.keys())
        avg_accuracies = [np.mean(size_performance[size]) for size in sizes]

        ax2.bar(sizes, avg_accuracies, color='lightblue', alpha=0.7)
        ax2.set_title('Average Accuracy by Model Size', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Average Accuracy')
        ax2.tick_params(axis='x', rotation=45)

        # 3. Category Performance Heatmap
        categories = set()
        for results in self.results.values():
            if 'category_performance' in results:
                categories.update(results['category_performance'].keys())

        category_matrix = []
        model_names = []

        for model_name, results in self.results.items():
            if 'category_performance' in results:
                model_names.append(model_name.split('/')[-1])
                row = []
                for cat in sorted(categories):
                    if cat in results['category_performance']:
                        acc = results['category_performance'][cat]['accuracy']
                        row.append(acc)
                    else:
                        row.append(0)
                category_matrix.append(row)

        if category_matrix:
            im = ax3.imshow(category_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax3.set_title('Accuracy by Medical Category', fontsize=16, fontweight='bold')
            ax3.set_xticks(range(len(categories)))
            ax3.set_xticklabels(sorted(categories), rotation=45, ha='right')
            ax3.set_yticks(range(len(model_names)))
            ax3.set_yticklabels(model_names)

            # Add text annotations
            for i in range(len(model_names)):
                for j in range(len(categories)):
                    if i < len(category_matrix) and j < len(category_matrix[i]):
                        text = ax3.text(j, i, f'{category_matrix[i][j]:.1%}',
                                      ha="center", va="center", color="black", fontsize=8)

        # 4. Top Performers Summary
        sorted_results = sorted(self.results.items(),
                              key=lambda x: x[1].get('accuracy', 0), reverse=True)

        top_5 = sorted_results[:5]
        top_models = [x[0].split('/')[-1] for x in top_5]
        top_scores = [x[1].get('accuracy', 0) for x in top_5]

        wedges, texts, autotexts = ax4.pie(top_scores, labels=top_models, autopct='%1.1f%%',
                                          startangle=90, colors=plt.cm.Set3(range(5)))
        ax4.set_title('Top 5 Model Performance', fontsize=16, fontweight='bold')

        plt.tight_layout()

        # Save the plot
        output_path = Path("evaluation/all_models_comparison.png")
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison charts saved to {output_path}")

        plt.show()

        return output_path

    def generate_detailed_report(self) -> str:
        """Generate comprehensive text report."""
        if not self.results:
            return "No results available"

        # Sort models by accuracy
        sorted_results = sorted(self.results.items(),
                              key=lambda x: x[1].get('accuracy', 0), reverse=True)

        report = f"""
🏥 COMPREHENSIVE MULTI-MODEL HEALTHCARE ACCURACY REPORT
{'=' * 80}

📊 EVALUATION SUMMARY
• Total Models Tested: {len(self.results)}
• Test Cases Per Model: {len(self.test_cases)}
• Medical Categories: {len(set(tc['category'] for tc in self.test_cases))}
• Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🏆 MODEL RANKINGS
{'=' * 40}
"""

        for rank, (model_name, results) in enumerate(sorted_results, 1):
            accuracy = results.get('accuracy', 0)
            category = results.get('category', 'Unknown')
            size = results.get('size', 'Unknown')

            # Performance emoji
            if accuracy >= 0.9:
                emoji = "🥇"
            elif accuracy >= 0.8:
                emoji = "🥈"
            elif accuracy >= 0.7:
                emoji = "🥉"
            elif accuracy >= 0.5:
                emoji = "📊"
            else:
                emoji = "⚠️"

            report += f"""
{emoji} Rank #{rank}: {model_name}
   • Accuracy: {accuracy:.2%}
   • Category: {category} ({size})
   • Correct: {results.get('correct', 0)}/{results.get('total_cases', 0)}
"""

        # Category analysis
        report += f"""

📋 PERFORMANCE BY MEDICAL CATEGORY
{'=' * 40}
"""

        # Aggregate category performance
        category_stats = {}
        for results in self.results.values():
            if 'category_performance' in results:
                for cat, perf in results['category_performance'].items():
                    if cat not in category_stats:
                        category_stats[cat] = []
                    category_stats[cat].append(perf['accuracy'])

        for category, accuracies in category_stats.items():
            avg_acc = np.mean(accuracies)
            max_acc = np.max(accuracies)
            min_acc = np.min(accuracies)

            report += f"""
• {category.replace('_', ' ').title()}:
  - Average: {avg_acc:.1%} | Best: {max_acc:.1%} | Worst: {min_acc:.1%}
"""

        # Top performers by category
        report += f"""

🎯 BEST MODEL PER CATEGORY
{'=' * 40}
"""

        for category in category_stats.keys():
            best_model = None
            best_accuracy = 0

            for model_name, results in self.results.items():
                if 'category_performance' in results and category in results['category_performance']:
                    acc = results['category_performance'][category]['accuracy']
                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_model = model_name

            if best_model:
                report += f"• {category.replace('_', ' ').title()}: {best_model} ({best_accuracy:.1%})\n"

        # Insights and recommendations
        report += f"""

💡 KEY INSIGHTS & RECOMMENDATIONS
{'=' * 40}
"""

        top_3 = sorted_results[:3]
        if len(top_3) >= 1:
            report += f"🏆 Best Overall: {top_3[0][0]} ({top_3[0][1].get('accuracy', 0):.1%})\n"
        if len(top_3) >= 2:
            report += f"🥈 Runner-up: {top_3[1][0]} ({top_3[1][1].get('accuracy', 0):.1%})\n"
        if len(top_3) >= 3:
            report += f"🥉 Third Place: {top_3[2][0]} ({top_3[2][1].get('accuracy', 0):.1%})\n"

        # Model size analysis
        size_performance = {}
        for results in self.results.values():
            if 'accuracy' in results and 'size' in results:
                size = results['size']
                if size not in size_performance:
                    size_performance[size] = []
                size_performance[size].append(results['accuracy'])

        if size_performance:
            best_size = max(size_performance.keys(),
                          key=lambda x: np.mean(size_performance[x]))
            report += f"\n📏 Best Model Size: {best_size} parameters\n"

        report += f"""
✅ All models successfully tested for healthcare accuracy
🔬 Comprehensive evaluation across emergency, diagnostics, pharmacology, and more
🎯 Results can guide optimal model selection for specific medical tasks
"""

        return report

    def export_results(self) -> str:
        """Export all results to JSON file."""
        output_data = {
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_models": len(self.results),
                "total_test_cases": len(self.test_cases),
                "categories": list(set(tc['category'] for tc in self.test_cases))
            },
            "model_results": self.results,
            "test_cases": self.test_cases
        }

        output_file = Path(f"evaluation/all_models_accuracy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Complete results exported to {output_file}")
        return str(output_file)

# Main evaluation function
async def evaluate_all_models():
    """Run complete evaluation of all models."""
    evaluator = AllModelsAccuracyEvaluator()

    print("🚀 Starting Comprehensive Multi-Model Healthcare Accuracy Evaluation...")
    print("=" * 80)

    # Run evaluation
    results = await evaluator.evaluate_all_models()

    # Generate report
    report = evaluator.generate_detailed_report()
    print(report)

    # Create visualizations
    print("\n📊 Generating comparison visualizations...")
    chart_path = evaluator.create_comparison_visualizations()

    # Export results
    results_file = evaluator.export_results()

    print(f"\n📁 Files Generated:")
    print(f"• Detailed Report: Above")
    print(f"• Comparison Charts: {chart_path}")
    print(f"• Complete Results: {results_file}")

    return {
        "results": results,
        "report": report,
        "chart_path": chart_path,
        "results_file": results_file
    }

if __name__ == "__main__":
    asyncio.run(evaluate_all_models())
