#!/bin/zsh
# Double-click (or `open play.command`) to start the neuron playground in Terminal.
cd "${0:A:h}"
source .venv/bin/activate
clear
echo "Neuron playground - LFM2.5-2.6B   (loading takes ~10 s)"
echo
echo "Try:  /cands            list the top 'hostile' neurons"
echo "      /try 5            steer candidate #5 and compare on 3 prompts"
echo "      /steer 16:1399:+0.5   then just type a message"
echo "      /compare Tell me about cats"
echo "      /help             all commands"
echo
python chat.py
