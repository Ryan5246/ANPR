from ultralytics import YOLO
import cv2
from paddleocr import TextRecognition

# Initialize the text recognition engine
ocr_engine = TextRecognition()
model = YOLO('runs/detect/train-3/weights/best.pt')

video_path = 'Testing Videos/trainingsvideo.avi'  
capture = cv2.VideoCapture(video_path)

if not capture.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

window_name = 'Optimized Video Feed - Full UI Controls'

# 1. Initialize the window first with resizable properties
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720) # Fits nicely on your display monitor

# --- TRACKBAR CALLBACKS ---
def on_timeline_change(val):
    current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    if abs(current_frame - val) > 2: 
        capture.set(cv2.CAP_PROP_POS_FRAMES, val)

def on_play_status_change(val):
    pass

# 2. Attach the UI Controls directly to the newly sized window
cv2.createTrackbar('Timeline', window_name, 0, total_frames, on_timeline_change)
cv2.createTrackbar('Play/Pause', window_name, 1, 1, on_play_status_change)

frame_count = 0
OCR_INTERVAL = 6  

while True:
    is_playing = cv2.getTrackbarPos('Play/Pause', window_name)
    
    if is_playing == 1:
        isTrue, frame = capture.read()
        if not isTrue:
            print("Video finished. Exiting...")
            break
            
        frame_count += 1
        
        # Update the timeline scrollbar marker position as the video plays
        current_frame_idx = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.setTrackbarPos('Timeline', window_name, current_frame_idx)
    else:
        # If paused, hold onto the frame targeted by the timeline slider
        current_frame_idx = cv2.getTrackbarPos('Timeline', window_name)
        capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        isTrue, frame = capture.read()
        if not isTrue:
            continue

    # Run YOLO matrix operations
    # Change this line inside your while loop:
    results = model(frame, imgsz=416, verbose=False, conf=0.65) 
    annotated_frame = frame.copy()
    
    for r in results:
        annotated_frame = r.plot() 
        
        # Interval check to skip heavy CPU text processing loops
        if is_playing == 1 and frame_count % OCR_INTERVAL == 0:
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                y1, y2 = max(0, y1), min(frame.shape[0], y2)
                x1, x2 = max(0, x1), min(frame.shape[1], x2)
                
                cropped_obj = frame[y1:y2, x1:x2]
                
                if cropped_obj.size > 0:
                    ocr_output = ocr_engine.predict(input=cropped_obj)
                    # NEW CODE
                    for res in ocr_output:
                        res_dict = dict(res)
                        # Access 'rec_text' directly from the top level of the converted dictionary
                        plate_text = res_dict['rec_text']
                        print(plate_text)
    # Render the optimized UI layout view
    cv2.imshow(window_name, annotated_frame)
    
    if cv2.waitKey(10) & 0xFF == ord('d'):
        break

capture.release()
cv2.destroyAllWindows()