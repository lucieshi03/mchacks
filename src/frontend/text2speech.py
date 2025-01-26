import pyttsx3

def speaker(msg):
    engine = pyttsx3.init()
    engine.say(msg)
    engine.runAndWait()


msg="Samosas are out here waging a personal war against my taste buds! Who decided they need to be this spicy? It's like biting into molten lava wrapped in dough. I wanted a snack, not a challenge to my existence. Why can't we enjoy the beautiful crunch and savory filling without setting our mouths on fire? Is it too much to ask for balance? Samosas, calm down with the spice—it’s supposed to be a treat, not a daredevil stunt for my tongue!"
speaker(msg)