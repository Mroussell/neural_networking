"""
MJR
Michael Roussell
Copyright 2022

Plase run from this main file!

Python 3.9.12 version of the python interpreter.
If there are any questions, please contact me at 'mjr.dev.contact@gmail.com.

MIT Education License Preferred.
"""
from perceptron import Perceptron

if __name__ == '__main__':
    run = True
    while(run):   
        x = int(input("Give value for x: (1 or 0)\n"))
        y = int(input("Give value for y: (1 or 0)\n"))

        perceptron = Perceptron()
        perceptron.fit_or(x=x,y=y)
        print(f"Prediction for OR: {perceptron.pred}")
        perceptron.fit_and(x=x,y=y)
        print(f"Prediction for AND: {perceptron.pred}")
        resp = None
        while(resp == None):
            resp = str(input("Would you like to run another test (y or n):\n"))
            if resp == 'y':
                continue
            elif resp == 'n':
                run = False
                break
            else:
                print(f"Please enter either 'y' or 'n'!")
                resp = None

        