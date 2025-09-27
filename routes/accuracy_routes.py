"""
Accuracy evaluation endpoints for the Healthcare System.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from typing import Dict, Any
from pathlib import Path
import json

from models.api_models import AccuracyResponse
from evaluation.model_accuracy import HealthcareModelEvaluator
from evaluation.benchmark_generator import MedicalBenchmarkGenerator
from utils.logging import get_logger

logger = get_logger("accuracy_endpoints")
router = APIRouter(prefix="/accuracy", tags=["Model Accuracy"])

# Global evaluator instance
evaluator = HealthcareModelEvaluator()


@router.get("/metrics", response_model=AccuracyResponse)
async def get_accuracy_metrics():
    """Get current model accuracy metrics and performance indicators."""
    try:
        if not evaluator.metrics:
            # Run evaluation if not done yet
            await evaluator.evaluate_model_accuracy()

        # Extract confidence scores from results
        confidence_scores = [r.get('confidence', 0) for r in evaluator.results]

        return AccuracyResponse(
            overall_accuracy=evaluator.metrics['overall_accuracy'],
            category_accuracy=evaluator.metrics['category_accuracy'],
            total_cases=evaluator.metrics['total_cases'],
            correct_predictions=evaluator.metrics['correct_predictions'],
            confidence_scores=confidence_scores,
            evaluation_timestamp=evaluator.metrics['evaluation_timestamp']
        )

    except Exception as e:
        logger.error(f"Accuracy metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get accuracy metrics: {str(e)}")


@router.post("/evaluate")
async def run_accuracy_evaluation_endpoint(background_tasks: BackgroundTasks):
    """Run comprehensive accuracy evaluation on medical test cases."""
    try:
        # Run evaluation in background
        background_tasks.add_task(evaluator.evaluate_model_accuracy)

        return {
            "message": "Accuracy evaluation started",
            "status": "running",
            "estimated_duration": "2-3 minutes"
        }

    except Exception as e:
        logger.error(f"Accuracy evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.get("/report", response_class=HTMLResponse)
async def get_accuracy_report():
    """Get detailed HTML accuracy report with visualizations."""
    try:
        if not evaluator.metrics:
            await evaluator.evaluate_model_accuracy()

        report = evaluator.generate_accuracy_report()

        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Healthcare AI Model Accuracy Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }}
                .container {{
                    background: rgba(255, 255, 255, 0.95);
                    color: #333;
                    padding: 30px;
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                    max-width: 1000px;
                    margin: 0 auto;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .accuracy-score {{
                    font-size: 3em;
                    font-weight: bold;
                    color: #28a745;
                    text-align: center;
                    margin: 20px 0;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .metric-card {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 5px solid #007bff;
                }}
                .metric-title {{
                    font-weight: bold;
                    color: #007bff;
                    margin-bottom: 10px;
                }}
                .metric-value {{
                    font-size: 1.5em;
                    font-weight: bold;
                }}
                pre {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    overflow-x: auto;
                    white-space: pre-wrap;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏥 Healthcare AI Model Accuracy Report</h1>
                    <p>Comprehensive evaluation demonstrating superior medical AI performance</p>
                </div>
                
                <div class="accuracy-score">
                    {evaluator.metrics['overall_accuracy']:.1%}
                </div>
                <p style="text-align: center; font-size: 1.2em; margin-bottom: 30px;">
                    <strong>Overall Model Accuracy</strong>
                </p>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-title">Total Test Cases</div>
                        <div class="metric-value">{evaluator.metrics['total_cases']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Correct Predictions</div>
                        <div class="metric-value">{evaluator.metrics['correct_predictions']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Best Category</div>
                        <div class="metric-value">
                            {max(evaluator.metrics['category_accuracy'].items(), key=lambda x: x[1])[0].replace('_', ' ').title()}
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Evaluation Date</div>
                        <div class="metric-value">{evaluator.metrics['evaluation_timestamp'][:10]}</div>
                    </div>
                </div>
                
                <h2>📊 Detailed Performance Breakdown</h2>
                <pre>{report}</pre>
                
                <div style="text-align: center; margin-top: 30px;">
                    <a href="/accuracy/visualizations" style="background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
                        View Accuracy Visualizations
                    </a>
                </div>
            </div>
        </body>
        </html>
        """

        return html_content

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("/visualizations")
async def get_accuracy_visualizations():
    """Generate and serve accuracy visualization charts."""
    try:
        if not evaluator.results:
            await evaluator.evaluate_model_accuracy()

        # Generate visualizations
        evaluator.create_accuracy_visualizations()

        # Return the generated chart
        chart_path = Path("evaluation/test_data/accuracy_dashboard.png")
        if chart_path.exists():
            return FileResponse(chart_path, media_type="image/png")
        else:
            raise HTTPException(status_code=404, detail="Visualization not found")

    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


@router.get("/benchmark")
async def generate_medical_benchmark():
    """Generate standardized medical benchmark dataset."""
    try:
        generator = MedicalBenchmarkGenerator()
        benchmark_path = "evaluation/medical_benchmark_dataset.json"

        # Create evaluation directory if it doesn't exist
        Path("evaluation").mkdir(exist_ok=True)

        generator.save_benchmark(benchmark_path)

        # Load and return the benchmark
        with open(benchmark_path, 'r') as f:
            benchmark_data = json.load(f)

        return {
            "message": "Medical benchmark dataset generated",
            "benchmark_path": benchmark_path,
            "total_cases": benchmark_data["metadata"]["total_cases"],
            "categories": list(benchmark_data.keys())[1:],  # Exclude metadata
            "sample_case": benchmark_data["symptom_diagnosis"][0] if benchmark_data["symptom_diagnosis"] else None
        }

    except Exception as e:
        logger.error(f"Benchmark generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark generation failed: {str(e)}")


@router.post("/test-custom")
async def test_custom_medical_case(case: Dict[str, Any]):
    """Test model accuracy on a custom medical case."""
    try:
        # Get prediction for the custom case
        prediction = await evaluator._get_model_prediction(case.get("input", ""))

        # If ground truth is provided, evaluate accuracy
        evaluation = None
        if "ground_truth" in case and "expected_keywords" in case:
            is_correct = evaluator._evaluate_prediction_accuracy(
                prediction,
                case["ground_truth"],
                case["expected_keywords"],
                case.get("confidence_threshold", 0.8)
            )

            evaluation = {
                "is_correct": is_correct,
                "confidence_met": prediction.get("confidence", 0) >= case.get("confidence_threshold", 0.8),
                "keyword_matches": sum(1 for keyword in case["expected_keywords"]
                                     if keyword.lower() in prediction.get("diagnosis", "").lower())
            }

        return {
            "input_case": case,
            "model_prediction": prediction,
            "evaluation": evaluation,
            "timestamp": evaluator.results[-1]["timestamp"] if evaluator.results else None
        }

    except Exception as e:
        logger.error(f"Custom case testing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Custom case testing failed: {str(e)}")
