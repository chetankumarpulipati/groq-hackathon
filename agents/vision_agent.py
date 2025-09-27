"""
Vision Processing Agent for analyzing medical images and visual data.
Optimized for healthcare imagery with high accuracy medical analysis.
"""

import asyncio
import base64
import io
import json
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pydicom
from agents.base_agent import BaseAgent
from config.settings import config
from utils.logging import get_logger
from utils.error_handling import VisionProcessingError, handle_exception

logger = get_logger("vision_agent")


class VisionAgent(BaseAgent):
    """
    Specialized agent for processing medical images and visual healthcare data.
    Uses advanced vision models for accurate medical image analysis.
    """

    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(agent_id, specialized_task="vision")
        self.supported_formats = config.vision.supported_formats
        self.max_image_size = config.vision.max_image_size
        self.medical_image_types = self._initialize_medical_image_types()

        logger.info(f"VisionAgent {self.agent_id} initialized for medical image analysis")

    def _initialize_medical_image_types(self) -> Dict[str, Dict[str, Any]]:
        """Initialize medical image type definitions and analysis parameters."""
        return {
            "xray": {
                "description": "X-ray radiographic images",
                "common_findings": ["fractures", "pneumonia", "cardiac_abnormalities", "bone_density"],
                "analysis_focus": "bone_structures_and_organs",
                "preprocessing": "contrast_enhancement"
            },
            "ct_scan": {
                "description": "Computed Tomography scans",
                "common_findings": ["tumors", "internal_bleeding", "organ_damage", "infections"],
                "analysis_focus": "soft_tissue_and_organs",
                "preprocessing": "windowing"
            },
            "mri": {
                "description": "Magnetic Resonance Imaging",
                "common_findings": ["brain_abnormalities", "spinal_issues", "soft_tissue_damage"],
                "analysis_focus": "soft_tissue_detail",
                "preprocessing": "noise_reduction"
            },
            "ultrasound": {
                "description": "Ultrasound imaging",
                "common_findings": ["pregnancy_monitoring", "organ_assessment", "blood_flow"],
                "analysis_focus": "real_time_imaging",
                "preprocessing": "speckle_reduction"
            },
            "dermoscopy": {
                "description": "Dermatoscopic images",
                "common_findings": ["skin_lesions", "moles", "melanoma_detection"],
                "analysis_focus": "skin_abnormalities",
                "preprocessing": "color_normalization"
            },
            "retinal": {
                "description": "Retinal fundus images",
                "common_findings": ["diabetic_retinopathy", "glaucoma", "macular_degeneration"],
                "analysis_focus": "retinal_blood_vessels",
                "preprocessing": "illumination_correction"
            }
        }

    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process medical image input data."""
        try:
            if isinstance(input_data, str):
                # File path processing
                return await self._process_image_file(input_data, context)
            elif isinstance(input_data, dict):
                if "image_path" in input_data:
                    return await self._process_image_with_metadata(input_data, context)
                elif "image_data" in input_data:
                    return await self._process_image_data(input_data, context)
                elif "dicom_path" in input_data:
                    return await self._process_dicom_file(input_data["dicom_path"], context)

            raise VisionProcessingError("Invalid input format for vision processing")

        except Exception as e:
            logger.error(f"Vision processing failed: {str(e)}")
            raise VisionProcessingError(f"Vision processing error: {str(e)}")

    @handle_exception
    async def _process_image_file(self, image_path: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a medical image file."""

        # Validate and load image
        image_info = await self._validate_and_load_image(image_path)

        # Preprocess image for medical analysis
        processed_image = await self._preprocess_medical_image(
            image_info["image"],
            image_info["detected_type"]
        )

        # Perform AI-based medical analysis
        medical_analysis = await self._perform_medical_image_analysis(
            processed_image,
            image_info["detected_type"],
            context
        )

        # Generate clinical report
        clinical_report = await self._generate_clinical_report(
            medical_analysis,
            image_info,
            context
        )

        return {
            "image_path": image_path,
            "image_info": image_info,
            "medical_analysis": medical_analysis,
            "clinical_report": clinical_report,
            "processing_type": "medical_image_file",
            "requires_radiologist_review": True
        }

    @handle_exception
    async def _process_image_with_metadata(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process image with additional medical metadata."""

        image_path = input_data["image_path"]
        patient_info = input_data.get("patient_info", {})
        clinical_context = input_data.get("clinical_context", "")
        image_type = input_data.get("image_type", "unknown")

        # Enhanced context with patient information
        enhanced_context = {
            **(context or {}),
            "patient_info": patient_info,
            "clinical_context": clinical_context,
            "specified_image_type": image_type
        }

        # Process image with enhanced context
        result = await self._process_image_file(image_path, enhanced_context)

        # Add metadata to result
        result.update({
            "patient_info": patient_info,
            "clinical_context": clinical_context,
            "processing_type": "image_with_metadata"
        })

        return result

    @handle_exception
    async def _process_dicom_file(self, dicom_path: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process DICOM medical image files."""

        try:
            # Read DICOM file
            dicom_data = pydicom.dcmread(dicom_path)

            # Extract DICOM metadata
            dicom_metadata = {
                "patient_id": getattr(dicom_data, 'PatientID', 'Unknown'),
                "study_date": getattr(dicom_data, 'StudyDate', 'Unknown'),
                "modality": getattr(dicom_data, 'Modality', 'Unknown'),
                "body_part": getattr(dicom_data, 'BodyPartExamined', 'Unknown'),
                "study_description": getattr(dicom_data, 'StudyDescription', 'Unknown'),
                "image_orientation": getattr(dicom_data, 'ImageOrientationPatient', []),
                "pixel_spacing": getattr(dicom_data, 'PixelSpacing', [])
            }

            # Convert DICOM to processable image
            image_array = dicom_data.pixel_array
            image = Image.fromarray(image_array)

            # Detect medical image type from DICOM metadata
            detected_type = self._detect_image_type_from_dicom(dicom_metadata)

            # Preprocess for medical analysis
            processed_image = await self._preprocess_medical_image(image, detected_type)

            # Perform medical analysis
            medical_analysis = await self._perform_medical_image_analysis(
                processed_image,
                detected_type,
                {**(context or {}), "dicom_metadata": dicom_metadata}
            )

            # Generate clinical report
            clinical_report = await self._generate_clinical_report(
                medical_analysis,
                {"detected_type": detected_type, "format": "dicom"},
                context
            )

            return {
                "dicom_path": dicom_path,
                "dicom_metadata": dicom_metadata,
                "detected_type": detected_type,
                "medical_analysis": medical_analysis,
                "clinical_report": clinical_report,
                "processing_type": "dicom_file",
                "requires_radiologist_review": True
            }

        except Exception as e:
            raise VisionProcessingError(f"DICOM processing failed: {str(e)}")

    async def _validate_and_load_image(self, image_path: str) -> Dict[str, Any]:
        """Validate and load medical image with metadata extraction."""

        image_path = Path(image_path)

        if not image_path.exists():
            raise VisionProcessingError(f"Image file not found: {image_path}")

        file_extension = image_path.suffix.lower().lstrip('.')
        if file_extension not in self.supported_formats:
            raise VisionProcessingError(f"Unsupported image format: {file_extension}")

        try:
            # Load image
            image = Image.open(image_path)

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Check image size
            if max(image.size) > self.max_image_size:
                # Resize while maintaining aspect ratio
                image.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)

            # Detect medical image type
            detected_type = await self._detect_medical_image_type(image, image_path.name)

            # Extract basic image metadata
            image_info = {
                "image": image,
                "file_path": str(image_path),
                "format": file_extension,
                "size": image.size,
                "mode": image.mode,
                "detected_type": detected_type,
                "file_size": image_path.stat().st_size
            }

            return image_info

        except Exception as e:
            raise VisionProcessingError(f"Image loading failed: {str(e)}")

    async def _detect_medical_image_type(self, image: Image.Image, filename: str) -> str:
        """Detect the type of medical image using AI analysis."""

        # Convert image to base64 for AI analysis
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Filename-based hints
        filename_lower = filename.lower()
        type_hints = []

        for img_type in self.medical_image_types.keys():
            if img_type.replace('_', '') in filename_lower:
                type_hints.append(img_type)

        # AI-based image type detection
        system_prompt = f"""You are a medical imaging specialist. Analyze this medical image and determine its type.
        
        Possible types: {', '.join(self.medical_image_types.keys())}
        Filename hints: {', '.join(type_hints) if type_hints else 'None'}
        
        Consider:
        - Image characteristics (contrast, texture, anatomical structures)
        - Typical imaging modality features
        - File naming patterns
        
        Respond with just the most likely image type from the list above."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this medical image. Filename: {filename}"}
        ]

        try:
            ai_response = await self.get_ai_response(messages, temperature=0.1)
            detected_type = ai_response["response"].strip().lower()

            # Validate detected type
            if detected_type in self.medical_image_types:
                return detected_type

        except Exception as e:
            logger.warning(f"AI-based type detection failed: {e}")

        # Fallback to filename-based detection
        if type_hints:
            return type_hints[0]

        return "unknown"

    def _detect_image_type_from_dicom(self, dicom_metadata: Dict[str, Any]) -> str:
        """Detect image type from DICOM metadata."""

        modality = dicom_metadata.get("modality", "").upper()
        body_part = dicom_metadata.get("body_part", "").lower()

        modality_mapping = {
            "CR": "xray",
            "DX": "xray",
            "CT": "ct_scan",
            "MR": "mri",
            "US": "ultrasound",
            "MG": "mammography"
        }

        detected_type = modality_mapping.get(modality, "unknown")

        # Refine based on body part
        if "retina" in body_part or "fundus" in body_part:
            detected_type = "retinal"
        elif "skin" in body_part or "dermato" in body_part:
            detected_type = "dermoscopy"

        return detected_type

    async def _preprocess_medical_image(self, image: Image.Image, image_type: str) -> Image.Image:
        """Preprocess medical image based on its type for optimal analysis."""

        if image_type not in self.medical_image_types:
            return image

        preprocessing_type = self.medical_image_types[image_type]["preprocessing"]

        try:
            if preprocessing_type == "contrast_enhancement":
                # Enhance contrast for X-rays
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5)

            elif preprocessing_type == "noise_reduction":
                # Apply noise reduction for MRI
                image_array = np.array(image)
                denoised = cv2.bilateralFilter(image_array, 9, 75, 75)
                image = Image.fromarray(denoised)

            elif preprocessing_type == "color_normalization":
                # Normalize colors for dermatoscopy
                image_array = np.array(image)
                normalized = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX)
                image = Image.fromarray(normalized)

            elif preprocessing_type == "illumination_correction":
                # Correct illumination for retinal images
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1.2)

            return image

        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image

    async def _perform_medical_image_analysis(
        self,
        image: Image.Image,
        image_type: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform comprehensive medical image analysis using AI."""

        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        image_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Get image type information
        image_type_info = self.medical_image_types.get(image_type, {})

        # Build comprehensive analysis prompt
        system_prompt = f"""You are an expert radiologist and medical imaging specialist. 
        
        Analyze this {image_type_info.get('description', 'medical image')} with focus on:
        - {image_type_info.get('analysis_focus', 'general medical findings')}
        - Common findings: {', '.join(image_type_info.get('common_findings', []))}
        
        Provide a comprehensive analysis including:
        1. Image quality assessment
        2. Anatomical structures visible
        3. Normal findings
        4. Abnormal findings (if any)
        5. Recommendations for further evaluation
        6. Confidence level of findings
        7. Critical/urgent findings requiring immediate attention
        
        Be precise, evidence-based, and include confidence scores for each finding.
        Always recommend physician review for definitive diagnosis."""

        # Add clinical context if available
        clinical_context = ""
        if context:
            patient_info = context.get("patient_info", {})
            clinical_context = context.get("clinical_context", "")
            dicom_metadata = context.get("dicom_metadata", {})

            if patient_info or clinical_context or dicom_metadata:
                system_prompt += f"\n\nAdditional Clinical Context:\n"
                if patient_info:
                    system_prompt += f"Patient Information: {patient_info}\n"
                if clinical_context:
                    system_prompt += f"Clinical Context: {clinical_context}\n"
                if dicom_metadata:
                    system_prompt += f"DICOM Metadata: {dicom_metadata}\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please analyze this {image_type} medical image thoroughly."}
        ]

        # Use the most accurate model for medical analysis
        ai_response = await self.get_ai_response(
            messages,
            temperature=0.1,  # Very low for medical accuracy
            provider_override=config.models.vision_provider
        )

        return {
            "analysis_text": ai_response["response"],
            "image_type": image_type,
            "image_type_info": image_type_info,
            "model_info": ai_response["model_info"],
            "clinical_context_used": clinical_context != "",
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _generate_clinical_report(
        self,
        medical_analysis: Dict[str, Any],
        image_info: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate structured clinical report from medical analysis."""

        system_prompt = """Generate a structured clinical radiology report based on the medical image analysis.
        
        Format the report with these sections:
        1. CLINICAL HISTORY (if available)
        2. TECHNIQUE 
        3. FINDINGS
        4. IMPRESSION
        5. RECOMMENDATIONS
        6. CRITICAL VALUES (if any)
        
        Use professional medical terminology and be concise yet comprehensive.
        Include confidence levels and always recommend physician correlation."""

        report_prompt = f"""
        Image Analysis: {medical_analysis['analysis_text']}
        Image Type: {medical_analysis['image_type']}
        Image Info: {image_info}
        Context: {context or 'None provided'}
        
        Generate a professional clinical report."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": report_prompt}
        ]

        ai_response = await self.get_ai_response(messages, temperature=0.2)

        return {
            "clinical_report": ai_response["response"],
            "report_type": "radiology_report",
            "generated_timestamp": datetime.utcnow().isoformat(),
            "requires_physician_review": True,
            "model_info": ai_response["model_info"]
        }

    async def get_vision_capabilities(self) -> Dict[str, Any]:
        """Get current vision processing capabilities."""

        return {
            "agent_id": self.agent_id,
            "supported_formats": self.supported_formats,
            "max_image_size": self.max_image_size,
            "medical_image_types": list(self.medical_image_types.keys()),
            "preprocessing_capabilities": [
                "contrast_enhancement", "noise_reduction",
                "color_normalization", "illumination_correction"
            ],
            "analysis_capabilities": [
                "anatomical_structure_detection", "abnormality_detection",
                "image_quality_assessment", "clinical_report_generation"
            ],
            "dicom_support": True,
            "ai_model_provider": config.models.vision_provider
        }
