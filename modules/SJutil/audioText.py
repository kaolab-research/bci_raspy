import typing
import textgrid
import pygame

class AudioClass:
    def __init__(self, pygame: pygame, volume = 0.7):

        self.mixer = pygame.mixer
        self.mixer.init()
        self.mixer.music.set_volume(volume)

        self.loaded = False
        

        # Read a TextGrid object from a file.
        tg = textgrid.TextGrid.fromFile(f'asset/audio/{"adollshouse"}.txt')
        self.transcript = tg[1]

    def load(self,fileName):
        """
            loads audio living inside asset/audio/
        """
        filePath = f'asset/audio/{fileName}.wav'
        self.mixer.music.load(filePath)
        self.loaded = True

    def play(self):
        if not self.loaded:
            raise Exception("no audio file is loaded")
        self.mixer.music.play() # non blocking play

        # hold until the audio is synced with transcript
        while True:
            currentTime = self.mixer.music.get_pos() / 1000
            if self.transcript.intervalContaining(currentTime) is not None:
                break
    
    def stop(self):
        self.mixer.music.stop()

    def getWord(self):
        # get word at this time of the audio file
        currentTime = self.mixer.music.get_pos() / 1000
        word = self.transcript.intervalContaining(currentTime).mark
        return word


if __name__ == "__main__":

    audio = AudioClass(pygame)
    name = "adollshouse"
    audio.load(name)
    audio.play()
    while True:
        print(audio.getWord())



