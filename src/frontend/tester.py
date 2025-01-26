import sound
import popup

blinkNow = True
badPosture = False

if badPosture == True:
    msg = "sit up straight betch"
    popup.show_info(msg)
    sound.ping()

if blinkNow == False:
    msg = "pls blinkity blink please"
    popup.show_info(msg)
    sound.ping()