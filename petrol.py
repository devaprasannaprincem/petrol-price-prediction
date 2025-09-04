import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import datetime

def generate_dataset(n=500):
	locations = ['Mumbai', 'Delhi', 'Chennai', 'Bangalore', 'Kolkata']
	petrol_types = ['Normal', 'Premium', 'ExtraPremium']
	data = []
	for _ in range(n):
		location = np.random.choice(locations)
		vehicle_capacity = np.random.randint(800, 3000)
		litres = np.random.randint(1, 60)
		petrol_type = np.random.choice(petrol_types)
		base_price = {'Normal': 106, 'Premium': 118, 'ExtraPremium': 130}[petrol_type]
		fluctuation = np.random.uniform(-5, 5)
		total_price = round((base_price + fluctuation) * litres, 2)
		data.append([location, vehicle_capacity, litres, petrol_type, total_price])
	df = pd.DataFrame(data, columns=['location', 'vehicle_capacity', 'litres', 'petrol_type', 'price'])
	return df

df = generate_dataset()
le_location = LabelEncoder()
le_petrol = LabelEncoder()
df['location'] = le_location.fit_transform(df['location'])
df['petrol_type'] = le_petrol.fit_transform(df['petrol_type'])
X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

class PetrolPriceApp:
	def __init__(self, root):
		self.root = root
		self.root.title("Petrol Price Predictor")
		self.root.geometry("600x500")
		self.root.configure(bg="#f0f0f0")
		self.history = []
		style = ttk.Style()
		style.theme_use("clam")
		style.configure("TButton", font=("Arial", 11), padding=6)
		style.configure("TLabel", font=("Arial", 11))
		style.configure("TCombobox", padding=5)
		self.setup_widgets()

	def setup_widgets(self):
		title = tk.Label(self.root, text="Petrol Price Predictor", font=("Helvetica", 18, "bold"),
						bg="#f0f0f0", fg="#2c3e50")
		title.pack(pady=10)

		frame = tk.Frame(self.root, bg="#f0f0f0")
		frame.pack(pady=10)

		ttk.Label(frame, text="Location:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
		self.location_cb = ttk.Combobox(frame, values=le_location.classes_.tolist(), state="readonly")
		self.location_cb.grid(row=0, column=1)

		ttk.Label(frame, text="Vehicle Capacity (cc):").grid(row=1, column=0, sticky="e", padx=5, pady=5)
		self.vehicle_cap_entry = ttk.Entry(frame)
		self.vehicle_cap_entry.grid(row=1, column=1)

		ttk.Label(frame, text="Litres:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
		self.litres_entry = ttk.Entry(frame)
		self.litres_entry.grid(row=2, column=1)

		ttk.Label(frame, text="Petrol Type:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
		self.petrol_cb = ttk.Combobox(frame, values=le_petrol.classes_.tolist(), state="readonly")
		self.petrol_cb.grid(row=3, column=1)

		btn_frame = tk.Frame(self.root, bg="#f0f0f0")
		btn_frame.pack(pady=10)
		ttk.Button(btn_frame, text="Predict Price", command=self.predict_price).grid(row=0, column=0, padx=10)
		ttk.Button(btn_frame, text="Clear", command=self.clear_form).grid(row=0, column=1, padx=10)
		ttk.Button(btn_frame, text="Save History", command=self.save_history).grid(row=0, column=2, padx=10)

		self.result_var = tk.StringVar()
		result_label = tk.Label(self.root, textvariable=self.result_var, font=("Helvetica", 16, "bold"),
							   bg="#f0f0f0", fg="#27ae60")
		result_label.pack(pady=10)

		self.history_box = tk.Text(self.root, height=10, width=70, state="disabled",
								  bg="#ffffff", fg="#2c3e50", font=("Courier", 10))
		self.history_box.pack(pady=5)

	def predict_price(self):
		try:
			loc = self.location_cb.get()
			cap = int(self.vehicle_cap_entry.get())
			litres = float(self.litres_entry.get())
			fuel = self.petrol_cb.get()
			if not loc or not fuel:
				raise ValueError("Location and petrol type must be selected.")
			loc_encoded = le_location.transform([loc])[0]
			fuel_encoded = le_petrol.transform([fuel])[0]
			input_data = pd.DataFrame([[loc_encoded, cap, litres, fuel_encoded]],
									 columns=['location', 'vehicle_capacity', 'litres', 'petrol_type'])
			prediction = model.predict(input_data)
			price = round(prediction[0][0], 2)
			self.result_var.set(f"Estimated Price: ₹{price:.2f}")
			log = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {loc}, {cap}cc, {litres}L, {fuel} => ₹{price}"
			self.update_history(log)
		except ValueError as ve:
			messagebox.showwarning("Input Error", str(ve))
		except Exception as e:
			messagebox.showerror("Error", f"An error occurred:\n{e}")

	def update_history(self, log):
		self.history.append(log)
		self.history_box.config(state="normal")
		self.history_box.insert("end", log + "\n")
		self.history_box.config(state="disabled")

	def clear_form(self):
		self.location_cb.set("")
		self.vehicle_cap_entry.delete(0, tk.END)
		self.litres_entry.delete(0, tk.END)
		self.petrol_cb.set("")
		self.result_var.set("")

	def save_history(self):
		if not self.history:
			messagebox.showinfo("No History", "No predictions to save.")
			return
		file_path = filedialog.asksaveasfilename(defaultextension=".csv",
												filetypes=[("CSV files", "*.csv")])
		if file_path:
			pd.DataFrame(self.history, columns=["Prediction Log"]).to_csv(file_path, index=False)
			messagebox.showinfo("Saved", f"History saved to {file_path}")

if __name__ == "__main__":
	root = tk.Tk()
	app = PetrolPriceApp(root)
	root.mainloop()