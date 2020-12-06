from azure.cognitiveservices.speech import AudioDataStream, SpeechRecognizer, SpeechConfig, SpeechSynthesizer, SpeechSynthesisOutputFormat
from azure.cognitiveservices.speech.audio import AudioOutputConfig
from azure.cognitiveservices.language.luis.authoring import LUISAuthoringClient
from azure.cognitiveservices.language.luis.runtime import LUISRuntimeClient
from msrest.authentication import CognitiveServicesCredentials
import json

luis_app_id = '20263b4d-b405-4c9b-8de8-e51663797c41' 
luis_key = 'b45490c8a83243f9a6320ec7e8e85a43'
luis_endpoint = 'https://koinonos-language-understanding.cognitiveservices.azure.com/'

# Configure speech recognizer
speech_key, service_region = "40a03ef9d3d44916bdcd1c4457b82c13", "eastus" 
speech_config = SpeechConfig(subscription=speech_key, region=service_region)
speech_recognizer = SpeechRecognizer(speech_config=speech_config)

# Configure speech synthesizer
audio_config = AudioOutputConfig(use_default_speaker=True)
synthesizer = SpeechSynthesizer(speech_config=speech_config)

runtimeCredentials = CognitiveServicesCredentials(luis_key)
clientRuntime = LUISRuntimeClient(endpoint=luis_endpoint, credentials=runtimeCredentials)

print("Start listening...")
speech = speech_recognizer.recognize_once()
try:   
    while speech.text != "Stop":
        # Production == slot name
        print("Your query is: ", speech.text)
        predictionRequest = { "query" : speech.text}

        predictionResponse = clientRuntime.prediction.get_slot_prediction(luis_app_id, "Production", predictionRequest)
        
        print("Top intent: {}".format(predictionResponse.prediction.top_intent))
        print("Sentiment: {}".format (predictionResponse.prediction.sentiment))
        print("Intents: ")

        for intent in predictionResponse.prediction.intents:
            print("\t{}".format (json.dumps(intent)))
        print("Entities: {}".format (predictionResponse.prediction.entities))
        synthesizer.speak_text_async("A simple test to write to a file.")
        # Use a one-time, synchronous call to transcribe the speech
        print("Start listening...")
        speech = speech_recognizer.recognize_once()
except Exception as ex:
    print(ex)
        
        