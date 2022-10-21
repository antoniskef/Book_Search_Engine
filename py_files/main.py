from elasticsearch import Elasticsearch
import neural
import gui
import tkinter as tk


if __name__ == '__main__':
    es = Elasticsearch(host="localhost", port=9200)

    t = tk.Tk()
    gui = gui.Gui(t, es)
    t.mainloop()









