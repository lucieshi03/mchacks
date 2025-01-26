import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox 


def show_info(msg):
    popup = tk.Tk()
    popup.title("HEY YOU")
    popup.geometry("950x450")
    popup.configure(bg="red")
    label = tk.Label(popup, text = msg, font=("Arial",25), bg="red", fg="white", wraplength=750)
    label.pack(pady=10)
    close_button = tk.Button(popup, text="OK", command=popup.destroy)
    close_button.pack(pady=10)
    popup.mainloop()
msg="Samosas are out here waging a personal war against my taste buds! Who decided they need to be this spicy? It's like biting into molten lava wrapped in dough. I wanted a snack, not a challenge to my existence. Why can't we enjoy the beautiful crunch and savory filling without setting our mouths on fire? Is it too much to ask for balance? Samosas, calm down with the spice—it’s supposed to be a treat, not a daredevil stunt for my tongue!"
show_info(msg)