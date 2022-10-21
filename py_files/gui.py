import tkinter as tk
from tkinter import ttk
import get
import add
import neural


WIDTH = 1300
HEIGHT = 700


class Gui:
    def __init__(self, t, es):
        t.title("Search Engine")
        screen_width = t.winfo_screenwidth()
        x_coordinate = int((screen_width / 2) - (WIDTH / 2))
        y_coordinate = 0
        t.geometry("{}x{}+{}+{}".format(WIDTH, HEIGHT, x_coordinate, y_coordinate))

        self.f1 = tk.Frame(t)
        self.f1.pack()

        self.f2 = tk.Frame(t)
        self.f2.pack()

        self.f3 = tk.Frame(t)
        self.f3.pack()

        self.w = tk.Label(self.f1, text="Book Search", font="Arial 25", fg='#a14f54')
        self.w.grid(row=0, columnspan=3, sticky="ew")

        self.w1 = tk.Label(self.f1, text="Enter UserID", font="Arial 17", fg='#a14f54')
        self.w1.grid(row=1, column=0, sticky="ew")

        # tk didn't work (because i am in a mac) - the entry didn't appear
        # in order to change the color due to ttk:
        ttk.Style().configure('black/white.TEntry', foreground='black', background='white')
        self.entry1 = ttk.Entry(self.f1, font='Arial 17', width=14, style='black/white.TEntry')
        self.entry1.grid(row=1, column=1, sticky="ew")

        self.button1 = tk.Button(self.f1, font='Arial 16', text="SAVE", command=self.pushed_id, fg='#a14f54')
        self.button1.grid(row=1, column=2, sticky="ew")

        self.w2 = tk.Label(self.f1, text="Search here", font="Arial 17", fg='#a14f54')
        self.w2.grid(row=2, column=0, sticky="ew")

        # tk didn't work (because i am in a mac) - the entry didn't appear
        # in order to change the color due to ttk:
        ttk.Style().configure('black/white.TEntry', foreground='black', background='white')
        self.entry2 = ttk.Entry(self.f1, font='Arial 17', width=14, style='black/white.TEntry')
        self.entry2.grid(row=2, column=1, sticky="ew")

        self.button2 = tk.Button(self.f1, font='Arial 16', text="SEARCH", command=self.pushed_search, fg='#a14f54')
        self.button2.grid(row=2, column=2, sticky="ew")

        self.es = es

        self.books = tk.Listbox(self.f3)

        self.scrollbar_y = tk.Scrollbar(self.f3, orient="vertical")
        self.scrollbar_x = tk.Scrollbar(self.f3, orient="horizontal")

        self.d = add.Add(self.es, '')
        self.d.add_average_ratings()

        self.id_pushed = 0

        self.text = tk.StringVar()
        self.w3 = tk.Label(self.f2, textvariable=self.text, font="Arial 17", fg='#a14f54')
        self.w3.grid(row=3, column=1)

    def pushed_id(self):
        self.id_pushed = 1

        self.d.userid = self.entry1.get()

        self.d.valid_user()

        self.text.set('Loading data...')
        self.f2.update()

        self.after_pushed_id()

    def after_pushed_id(self):
        self.d.add_user_ratings()

        self.d.valid_user()
        self.d.user_with_review()

        if self.d.with_review:
            n = neural.Neural()
            n.neural()

            self.d.add_elastic()
            self.text.set("Valid user that has made at least one review. The search results will be personalized.")
            self.f2.update()
        elif self.d.valid:
            self.d.add_elastic()
            self.text.set("Valid user but without any review. The search results will not be personalized.")
            self.f2.update()
        else:
            self.d.add_elastic()
            self.text.set("Not valid user. The search results will not be personalized.")
            self.f2.update()

    def pushed_search(self):
        if self.id_pushed:
            search = self.entry2.get()

            g = get.Get(self.es, search)
            g.get()

            self.books.destroy()
            self.books = tk.Listbox(self.f3, yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)
            self.books.config(width=WIDTH // 10, height=HEIGHT // 10)

            for hit in g.res['hits']['hits']:
                self.books.insert(tk.END, hit["_source"])

            self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
            self.scrollbar_y.config(command=self.books.yview)
            self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
            self.scrollbar_x.config(command=self.books.xview)

            self.books.pack(side=tk.LEFT, fill=tk.BOTH)

        else:
            self.text.set("Enter user ID first.")
            self.f2.update()



