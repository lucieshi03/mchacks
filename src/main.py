import frontend.sounds
import frontend.popup
from eyes.blink_eyes import check_blink_timeout
#testing

# Sounds
#if screen2close == 
#    sound.ping()
badPosture = True
blinkNow = check_blink_timeout() #call a function in blink_eyes to check if the user blinked

if badPosture == True:
    msg = "sit up straight betch"
    popup.show_info(msg)
    sound.ping()

if blinkNow == False:
    msg = "pls blinkity blink please"
    popup.show_info(msg)
    sound.ping()