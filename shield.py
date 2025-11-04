# Dr. Strange Shields - Gesture Control System
# Author: Soumya Suman Kar
# Copyright (c) 2025 Soumya Suman Kar
# Licensed under MIT License

# libraries
import cv2
import time
import mediapipe as mp
import numpy as np
import os
from utils import mediapipe_detection, get_center_lh,get_center_rh, points_detection, points_detection_hands, draw_landmarks_custom, draw_limit_lh, draw_limit_rh
from argparse import ArgumentParser
import pickle
from datetime import datetime, timedelta
import time
import pyvirtualcam
from pyvirtualcam import PixelFormat
import signal
import sys
import pygame

# - INPUT PARAMETERS ------------------------------- #
parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="ML_model", default='models/model_svm.sav',
                    help="PATH of model FILE.", metavar="FILE")
parser.add_argument("-t", "--threshold", dest="threshold_prediction", default=0.9, type=float,
                    help="Threshold for prediction. A number between 0 and 1. default is 0.5")
parser.add_argument("-dc", "--det_conf", dest="min_detection_confidence", default=0.5, type=float,
                    help="Threshold for prediction. A number between 0 and 1. default is 0.5")
parser.add_argument("-tc", "--trk_conf", dest="min_tracking_confidence", default=0.5, type=float,
                    help="Threshold for prediction. A number between 0 and 1. default is 0.5")
parser.add_argument("-c", "--camera_id", dest="camera", default=0, type=int,
                    help="ID of the camera. An integer between 0 and N. Default is 1")
parser.add_argument("-s", "--shield", dest="shield_video", default='effects/shield.mp4',
                    help="PATH of the video FILE.", metavar="FILE")
parser.add_argument("-o", "--output", dest="output_mode", default='both',
                    choices=['window', 'virtual', 'both'],
                    help="Output mode: 'window' for OpenCV window only, 'virtual' for virtual camera only, 'both' for both outputs. Default is 'both'")

args = parser.parse_args()
# -------------------------------------------------- #

# Variabili globali per la gestione della chiusura
cap = None
cam = None
show_window = False

def signal_handler(sig, frame):
    """Gestisce l'interruzione Ctrl+C per una chiusura pulita"""
    print("\n\n" + "="*60)
    print("\n🛑 Interruzione ricevuta (Ctrl+C)")
    print("🧹 Pulizia delle risorse in corso...")

    # Cleanup delle risorse
    if cap:
        cap.release()
        print("  ✅ Camera rilasciata")

    if show_window:
        cv2.destroyAllWindows()
        print("  ✅ Finestre OpenCV chiuse")

    if cam:
        cam.close()
        print("  ✅ Virtual camera chiusa")

    print("\n🏁 Applicazione terminata correttamente\n")
    print("="*60)
    sys.exit(0)

# Registra il gestore del segnale Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# Initialize pygame mixer for audio
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Function to generate audio effects
def generate_activation_sound():
    """Generate shield activation sound effect"""
    duration = 0.5  # seconds
    sample_rate = 22050
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Create a rising frequency effect
    frequency_sweep = np.linspace(200, 800, len(t))
    wave = np.sin(2 * np.pi * frequency_sweep * t)
    
    # Apply envelope for fade in/out
    envelope = np.exp(-3 * t / duration)
    wave = wave * envelope
    
    # Normalize and convert to 16-bit
    wave = np.int16(wave * 32767 / np.max(np.abs(wave)))
    
    # Convert to stereo
    stereo_wave = np.column_stack((wave, wave))
    return pygame.sndarray.make_sound(stereo_wave)

def generate_deactivation_sound():
    """Generate shield deactivation sound effect"""
    duration = 0.3  # seconds
    sample_rate = 22050
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Create a falling frequency effect
    frequency_sweep = np.linspace(800, 200, len(t))
    wave = np.sin(2 * np.pi * frequency_sweep * t)
    
    # Apply envelope
    envelope = np.exp(-5 * t / duration)
    wave = wave * envelope
    
    # Normalize and convert to 16-bit
    wave = np.int16(wave * 32767 / np.max(np.abs(wave)))
    
    # Convert to stereo
    stereo_wave = np.column_stack((wave, wave))
    return pygame.sndarray.make_sound(stereo_wave)

def generate_key_sound():
    """Generate key detection sound effect"""
    duration = 0.15  # seconds
    sample_rate = 22050
    frequency = 600
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = np.sin(2 * np.pi * frequency * t)
    
    # Apply envelope
    envelope = np.exp(-10 * t / duration)
    wave = wave * envelope
    
    # Normalize and convert to 16-bit
    wave = np.int16(wave * 32767 / np.max(np.abs(wave)))
    
    # Convert to stereo
    stereo_wave = np.column_stack((wave, wave))
    return pygame.sndarray.make_sound(stereo_wave)

# Generate sound effects
try:
    activation_sound = generate_activation_sound()
    deactivation_sound = generate_deactivation_sound()
    key_sound = generate_key_sound()
    audio_enabled = True
    print("♫ Audio effects loaded successfully")
except Exception as e:
    print(f"⚠️  Audio initialization failed: {e}")
    audio_enabled = False

current_directory = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")

# --- start camera ---
cap = cv2.VideoCapture(args.camera)
time.sleep(5)

# --- load svm model ---
model = pickle.load(open(current_directory + '/' + args.ML_model, 'rb'))
labels = np.array(model.classes_)  # put the entire alphabet in the future

KEY_1 = False
KEY_2 = False
KEY_3 = False
SHIELDS = False

scale = 1.5

mp_holistic = mp.solutions.holistic
#mp_drawing = mp.solutions.drawing_utils

# --- load shield video ---
shield = cv2.VideoCapture(current_directory + '/' + args.shield_video)

# --- get width and height from camera
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

black_screen = np.array([0,0,0])

# Verifica modalità di output
show_window = args.output_mode in ['window', 'both']
use_virtual_cam = args.output_mode in ['virtual', 'both']

print("\n" + "="*60)
print("🛡️  DR. STRANGE SHIELDS - GESTURE CONTROL SYSTEM 🛡️")
print("="*60)
print(f"\n📹 Camera ID: {args.camera}")
print(f"🤖 ML Model: {args.ML_model}")
print(f"🎯 Prediction Threshold: {args.threshold_prediction}")
print(f"🔍 Detection Confidence: {args.min_detection_confidence}")
print(f"📊 Tracking Confidence: {args.min_tracking_confidence}")
print(f"🎬 Shield Video: {args.shield_video}\n")
print("-" * 60)
print(f"\n📺 Output Mode: {args.output_mode.upper()}\n")
if show_window:
    print("  ✅ OpenCV Window: ENABLED")
else:
    print("  ❌ OpenCV Window: DISABLED")
if use_virtual_cam:
    print("  ✅ Virtual Camera: ENABLED\n")
else:
    print("  ❌ Virtual Camera: DISABLED\n")
print("-" * 60)

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5,
                          model_complexity=0) as holistic:

    # Inizializza la virtual camera solo se necessario
    if use_virtual_cam:
        cam = pyvirtualcam.Camera(width, height, 30, fmt=PixelFormat.BGR)
        print(f"\n🎥 Virtual Camera Device: {cam.device}")
        print("🚀 System Ready! Starting gesture detection...")
        print("📋 Gesture Sequence: KEY_1 → KEY_2 → KEY_3 (activate shields)")
        print("📋 Shield Deactivation: KEY_4")
        print("⌨️  Press 'q' to quit" + (" (in OpenCV window)" if show_window else "") + " or Ctrl+C\n")
        print("="*60 + "\n")
    else:
        cam = None
        print("🚀 System Ready! Starting gesture detection...")
        print("📋 Gesture Sequence: KEY_1 → KEY_2 → KEY_3 (activate shields)")
        print("📋 Shield Deactivation: KEY_4")
        print("⌨️  Press 'q' to quit" + (" (in OpenCV window)" if show_window else "") + " or Ctrl+C")
        print("="*60 + "\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            # Status display con formato migliorato
            status_shields = "🛡️ ON " if SHIELDS else "🛡️ OFF"
            status_k1 = "🔑1✅" if KEY_1 else "🔑1❌"
            status_k2 = "🔑2✅" if KEY_2 else "🔑2❌"
            status_k3 = "🔑3✅" if KEY_3 else "🔑3❌"

            print(f"\r{status_shields} | {status_k1} {status_k2} {status_k3}", end="", flush=True)

            ret_shield, frame_shield = shield.read()
            if not ret_shield:
                shield = cv2.VideoCapture(current_directory + '/' + args.shield_video)
                ret_shield, frame_shield = shield.read()

            # make detection
            frame, results = mediapipe_detection(frame, holistic)
            
            # Draw hand landmarks for visual feedback
            if results.left_hand_landmarks or results.right_hand_landmarks:
                draw_landmarks_custom(frame, results)
                draw_limit_lh(frame, results)
                draw_limit_rh(frame, results)
            
            # Add status text on screen
            status_text = f"Shields: {'ON' if SHIELDS else 'OFF'}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                       (0, 255, 0) if SHIELDS else (0, 0, 255), 2)
            
            gesture_status = f"Keys: {'1' if KEY_1 else '-'} {'2' if KEY_2 else '-'} {'3' if KEY_3 else '-'}"
            cv2.putText(frame, gesture_status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show instruction
            if not SHIELDS:
                cv2.putText(frame, "Show both hands and perform gestures", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            xMinL, xMaxL, yMinL, yMaxL = get_center_lh(frame, results)
            xMinR, xMaxR, yMinR, yMaxR = get_center_rh(frame, results)

            #hsv = cv2.cvtColor(frame_shield, cv2.COLOR_BGR2HSV)
            # black_screen = np.array([0,0,0])
            mask = cv2.inRange(frame_shield,black_screen,black_screen)
            res = cv2.bitwise_and(frame_shield, frame_shield, mask=mask)
            res = frame_shield - res
            alpha=1

            if SHIELDS and xMinL:

                xc_lh = (xMaxL+xMinL)/2
                yc_lh = (yMaxL+yMinL)/2
                xc_lh = int(width*xc_lh)
                yc_lh = int(height*yc_lh)

                l_width_shield = int(width*(xMaxL-xMinL)/2*3.5*scale)
                l_height_shield = int(height*(yMaxL-yMinL)/2*2*scale)

                res2 = cv2.resize(res, (l_width_shield*2, l_height_shield*2))

                start_h = 0
                start_w = 0
                stop_h = l_height_shield*2
                stop_w = l_width_shield*2

                f_start_h = yc_lh-l_height_shield
                f_stop_h = yc_lh+l_height_shield
                f_start_w = xc_lh-l_width_shield
                f_stop_w = xc_lh+l_width_shield

                if yc_lh-l_height_shield < 0:
                    start_h = -yc_lh+l_height_shield
                    f_start_h = 0
                if yc_lh+l_height_shield > height:
                    stop_h = l_height_shield + height - yc_lh
                    f_stop_h = height
                if xc_lh-l_width_shield < 0:
                    start_w = -xc_lh+l_width_shield
                    f_start_w = 0
                if xc_lh+l_width_shield > width:
                    stop_w = l_width_shield + width - xc_lh
                    f_stop_w = width

                res2 = res2[start_h:stop_h, start_w:stop_w,:]

                frame_shield =cv2.addWeighted(frame[f_start_h:f_stop_h,f_start_w:f_stop_w], alpha, res2, 1,1, frame)
                frame[f_start_h:f_stop_h,f_start_w:f_stop_w] = frame_shield

            if SHIELDS and xMinR:

                xc_rh = (xMaxR+xMinR)/2
                yc_rh = (yMaxR+yMinR)/2
                xc_rh = int(width*xc_rh)
                yc_rh = int(height*yc_rh)

                r_width_shield = int(width*(xMaxR-xMinR)/2*3.5*scale)
                r_height_shield = int(height*(yMaxR-yMinR)/2*2*scale)

                res3 = cv2.resize(res, (r_width_shield*2, r_height_shield*2))

                start_h = 0
                start_w = 0
                stop_h = r_height_shield*2
                stop_w = r_width_shield*2

                f_start_h = yc_rh-r_height_shield
                f_stop_h = yc_rh+r_height_shield
                f_start_w = xc_rh-r_width_shield
                f_stop_w = xc_rh+r_width_shield

                if yc_rh-r_height_shield < 0:
                    start_h = -yc_rh+r_height_shield
                    f_start_h = 0
                if yc_rh+r_height_shield > height:
                    stop_h = r_height_shield + height - yc_rh
                    f_stop_h = height
                if xc_rh-r_width_shield < 0:
                    start_w = -xc_rh+r_width_shield
                    f_start_w = 0
                if xc_rh+r_width_shield > width:
                    stop_w = r_width_shield + width - xc_rh
                    f_stop_w = width

                res3 = res3[start_h:stop_h, start_w:stop_w,:]
                frame_shield =cv2.addWeighted(frame[f_start_h:f_stop_h,f_start_w:f_stop_w], alpha, res3, 1,1, frame)
                frame[f_start_h:f_stop_h,f_start_w:f_stop_w] = frame_shield

            if xMinL and xMinR and SHIELDS:
                prediction = model.predict(np.array([points_detection_hands(results)]))[0]
                pred_prob = np.max(model.predict_proba(np.array([points_detection_hands(results)])))
                
                # Show what gesture is detected
                gesture_text = f"Detected: {prediction.upper()} ({pred_prob:.2f})"
                cv2.putText(frame, gesture_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (0, 255, 255), 2)

                if (prediction == 'key_4') and (pred_prob > 0.7):  # Lowered from 0.85
                    KEY_1 = False
                    KEY_2 = False
                    KEY_3 = False
                    SHIELDS = False
                    if audio_enabled:
                        try:
                            deactivation_sound.play()
                            print("\n🔊 SHIELD DEACTIVATION sound played!")
                        except Exception as e:
                            print(f"\n⚠️ Audio error: {e}")

            elif xMinL and xMinR and (not SHIELDS):
                prediction = model.predict(np.array([points_detection_hands(results)]))[0]
                pred_prob = np.max(model.predict_proba(np.array([points_detection_hands(results)])))
                
                # Show what gesture is detected and confidence
                gesture_text = f"Detected: {prediction.upper()} ({pred_prob:.2f})"
                color = (0, 255, 0) if pred_prob > 0.7 else (0, 165, 255)  # Green if confident, orange if not
                cv2.putText(frame, gesture_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, color, 2)
                
                # Show next required gesture
                if not KEY_1:
                    cv2.putText(frame, "Next: Perform KEY_1 gesture", (10, height - 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                elif KEY_1 and not KEY_2:
                    cv2.putText(frame, "Next: Perform KEY_2 gesture (2s left)", (10, height - 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                elif KEY_1 and KEY_2 and not KEY_3:
                    cv2.putText(frame, "Next: Perform KEY_3 gesture (2s left)", (10, height - 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                if (prediction == 'key_1') and (pred_prob > 0.7):  # Lowered from 0.85
                    t1 = datetime.now()
                    KEY_1 = True
                    if audio_enabled:
                        try:
                            key_sound.play()
                            print("\n🔊 KEY_1 sound played!")
                        except Exception as e:
                            print(f"\n⚠️ Audio error: {e}")
                elif (prediction == 'key_2') and (pred_prob > 0.7) and KEY_1:  # Lowered from 0.85
                    t2 = datetime.now()
                    if t1 + timedelta(seconds=2) > t2:
                        KEY_2 = True
                        if audio_enabled:
                            try:
                                key_sound.play()
                                print("\n🔊 KEY_2 sound played!")
                            except Exception as e:
                                print(f"\n⚠️ Audio error: {e}")
                    else:
                        KEY_1 = False
                        KEY_2 = False

                elif (prediction == 'key_3') and (pred_prob > 0.7) and KEY_1 and KEY_2:  # Lowered from 0.85
                    t3 =datetime.now()
                    if t2 + timedelta(seconds=2) > t3:
                        KEY_3 = True
                        SHIELDS = True
                        if audio_enabled:
                            try:
                                activation_sound.play()
                                print("\n🔊 SHIELD ACTIVATION sound played!")
                            except Exception as e:
                                print(f"\n⚠️ Audio error: {e}")
                    else:
                        KEY_1 = False
                        KEY_2 = False

            # Mostra la finestra OpenCV solo se richiesto
            if show_window:
                cv2.imshow('Dr. Strange shields', frame)

            # Invia alla virtual camera solo se richiesto
            if use_virtual_cam and cam:
                cam.send(frame)
                cam.sleep_until_next_frame()

            # Gestione tasti solo se la finestra è attiva
            if show_window and cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif not show_window:
                # Se non c'è finestra, usa un piccolo delay per evitare loop troppo veloce
                time.sleep(0.033)  # ~30 FPS

    except KeyboardInterrupt:
        # Gestione alternativa del Ctrl+C (nel caso il signal handler non funzioni)
        print("\n\n🛑 Interruzione ricevuta - Chiusura in corso...")
    except Exception as e:
        print(f"\n❌ Errore durante l'esecuzione: {e}")
    finally:
        # Cleanup finale - assicura che tutto venga pulito correttamente
        print("\n\n" + "="*60)
        print("\n🧹 Cleanup finale...\n")
        if cap:
            cap.release()
            print("  ✅ Camera rilasciata")
        if show_window:
            cv2.destroyAllWindows()
            print("  ✅ Finestre OpenCV chiuse")
        if cam:
            cam.close()
            print("  ✅ Virtual camera chiusa")
        print("\n🏁 Applicazione terminata\n")
