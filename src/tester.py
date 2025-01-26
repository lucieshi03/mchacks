from frontend.sound import ping
from frontend.popup import show_info
from gemini.get_msg import get_message
from frontend.text2speech import speaker
import threading

def bad_postures():
    ping()

    message = get_message()
    
    show_info_thread = threading.Thread(target=show_info, args=(message,),daemon=True)
    speaker_thread = threading.Thread(target=speaker, args=(message,))
    
    show_info_thread.start()
    speaker_thread.start()

    

def bad_blinks():
    ping()
    message = get_message()
    
    show_info_thread = threading.Thread(target=show_info, args=(message,))
    speaker_thread = threading.Thread(target=speaker, args=(message,))
    
    show_info_thread.start()
    speaker_thread.start()

bad_postures()