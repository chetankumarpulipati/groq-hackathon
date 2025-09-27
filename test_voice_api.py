"""
Test script to send audio/text to the voice processing API
"""

import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"
VOICE_ENDPOINT = f"{BASE_URL}/voice/process"

def test_voice_processing_with_text():
    """Test voice processing with text input"""
    print("🎤 Testing Voice Processing API with text input...")

    # Test data
    data = {
        "text": "Patient reports severe chest pain and difficulty breathing for the past 2 hours",
        "patient_id": "TEST_P001"
    }

    try:
        response = requests.post(VOICE_ENDPOINT, data=data)

        if response.status_code == 200:
            result = response.json()
            print("✅ Voice processing successful!")
            print(f"📝 Transcription: {result.get('transcription')}")
            print(f"😊 Sentiment: {result.get('sentiment')} (confidence: {result.get('sentiment_confidence', 0):.2f})")
            print(f"😰 Stress Level: {result.get('stress_level')} (confidence: {result.get('stress_confidence', 0):.2f})")
            print(f"🏥 Medical Indicators: {result.get('medical_indicators', [])}")
            print(f"💡 Recommendations: {result.get('recommendations')}")
            return result
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Server is not running or not accessible")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_voice_processing_with_audio():
    """Test voice processing with simulated audio file"""
    print("\n🎵 Testing Voice Processing API with audio file...")

    # Create a dummy audio file for testing
    audio_content = b"fake_audio_data_for_testing"

    files = {
        'audio': ('test_audio.wav', audio_content, 'audio/wav')
    }

    data = {
        'patient_id': 'TEST_P002'
    }

    try:
        response = requests.post(VOICE_ENDPOINT, files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            print("✅ Audio processing successful!")
            print(f"📝 Transcription: {result.get('transcription')}")
            print(f"😊 Sentiment: {result.get('sentiment')} (confidence: {result.get('sentiment_confidence', 0):.2f})")
            print(f"😰 Stress Level: {result.get('stress_level')} (confidence: {result.get('stress_confidence', 0):.2f})")
            print(f"🏥 Medical Indicators: {result.get('medical_indicators', [])}")
            print(f"💡 Recommendations: {result.get('recommendations')}")
            return result
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Server is not running or not accessible")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def check_server_health():
    """Check if the server is running"""
    print("🔍 Checking server health...")

    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Server is healthy!")
            print(f"📊 Status: {health_data.get('status')}")
            print(f"🏥 Service: {health_data.get('service')}")
            print(f"📅 Version: {health_data.get('version')}")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running")
        return False

if __name__ == "__main__":
    print("🏥 Healthcare Voice Processing API Test")
    print("=" * 50)

    # Check server health first
    if check_server_health():
        print("\n" + "=" * 50)

        # Test text input
        text_result = test_voice_processing_with_text()

        # Test audio input
        audio_result = test_voice_processing_with_audio()

        print("\n" + "=" * 50)
        print("🎯 Test Summary:")
        print(f"✅ Text Processing: {'Success' if text_result else 'Failed'}")
        print(f"✅ Audio Processing: {'Success' if audio_result else 'Failed'}")

    else:
        print("\n❌ Cannot run tests - server is not available")
        print("💡 Please start the server first: python main.py")
