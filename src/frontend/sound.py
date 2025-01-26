def ping():
    import winsound
    frequency = 2500
    duration = 1000
    winsound.Beep(frequency, duration)
    winsound.Beep(frequency, duration)
    winsound.Beep(frequency, duration)

ping()