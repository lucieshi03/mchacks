import pyttsx3
from pyttsx3 import *

def speaker(msg):
    engine = pyttsx3.init()

    rate = engine.getProperty('rate')   
    print (rate)                        
    engine.setProperty('rate', 225)

    engine.say(msg)
    engine.runAndWait()


msg="Honey, your posture screams 'I haven't seen a spine doctor since the Jurassic period,' and your blink rate suggests you're less human and more... taxidermied owl.  Ten slumps? Twenty missed blinks?  Sounds like your desk isn't the only thing that's stiff. You're practically a human statue, a monument to bad habits.  Maybe if you moved around a bit, you wouldn't look like you're already fossilizing.  Get a life, and a chiropractor!"
speaker(msg)