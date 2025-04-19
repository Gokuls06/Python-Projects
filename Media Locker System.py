import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Function to generate a key from a password using PBKDF2
def generate_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(password.encode())

# Function to encrypt a file
def encrypt_file(input_filename: str, password: str, output_filename: str):
    try:
        with open(input_filename, 'rb') as f:
            file_data = f.read()
        
        salt = os.urandom(16)
        key = generate_key(password, salt)

        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(file_data) + padder.finalize()

        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

        with open(output_filename, 'wb') as f:
            f.write(salt)
            f.write(iv)
            f.write(encrypted_data)

        messagebox.showinfo("Success", f"File encrypted successfully and saved as {output_filename}")

    except Exception as e:
        messagebox.showerror("Error", f"Encryption failed: {e}")

# Function to decrypt a file
def decrypt_file(input_filename: str, password: str, output_filename: str):
    try:
        with open(input_filename, 'rb') as f:
            salt = f.read(16)  # First 16 bytes are the salt
            iv = f.read(16)    # Next 16 bytes are the IV
            encrypted_data = f.read()

        key = generate_key(password, salt)

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()

        decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()

        unpadder = padding.PKCS7(128).unpadder()
        original_data = unpadder.update(decrypted_data) + unpadder.finalize()

        with open(output_filename, 'wb') as f:
            f.write(original_data)

        messagebox.showinfo("Success", f"File decrypted successfully and saved as {output_filename}")

        # Auto-open the decrypted file
        webbrowser.open(output_filename)

    except ValueError:
        messagebox.showerror("Error", "Decryption failed. Possible incorrect password.")
    except FileNotFoundError:
        messagebox.showerror("Error", f"File {input_filename} not found.")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

# GUI Functions
def select_file_to_encrypt():
    input_file = filedialog.askopenfilename(title="Select a file to encrypt")
    if not input_file:
        messagebox.showwarning("File Missing", "Please select a file to encrypt.")
        return

    output_file = filedialog.asksaveasfilename(defaultextension=".enc", filetypes=[("Encrypted Files", "*.enc")], title="Save Encrypted File As")
    if not output_file:
        messagebox.showwarning("Save Location Missing", "Please choose a location to save the encrypted file.")
        return

    password = password_entry.get()
    if not password or len(password) < 6:
        messagebox.showwarning("Password Missing", "Please enter a password of at least 6 characters.")
        return

    encrypt_file(input_file, password, output_file)

def select_file_to_decrypt():
    input_file = filedialog.askopenfilename(title="Select a file to decrypt")
    if not input_file:
        messagebox.showwarning("File Missing", "Please select a file to decrypt.")
        return

    output_file = filedialog.asksaveasfilename(defaultextension=".dec", filetypes=[("Decrypted Files", "*.dec")], title="Save Decrypted File As")
    if not output_file:
        messagebox.showwarning("Save Location Missing", "Please choose a location to save the decrypted file.")
        return

    password = password_entry.get()
    if not password or len(password) < 6:
        messagebox.showwarning("Password Missing", "Please enter a password of at least 6 characters.")
        return

    decrypt_file(input_file, password, output_file)

# Toggle password visibility
def toggle_password():
    if password_entry.cget("show") == "*":
        password_entry.config(show="")
        show_password_button.config(text="🙈 Hide")
    else:
        password_entry.config(show="*")
        show_password_button.config(text="👁 Show")

# Creating the main window for the app
root = tk.Tk()
root.title("🔒 Media Encryptor System")
root.geometry("420x380")
root.resizable(False, False)

# Use ttk for modern styling
style = ttk.Style()
style.configure("TButton", font=("Times New Roman", 12), padding=6)
style.configure("TLabel", font=("Times New Roman", 14))
style.configure("TEntry", font=("Aerial", 14), padding=6)

# Main Frame
frame = ttk.Frame(root, padding=25)
frame.pack(expand=True)

# Title Label
title_label = ttk.Label(frame, text="🔒 Media Encryptor System", font=("Times New Roman", 16, ""))
title_label.pack(pady=10)

# Password Entry
password_label = ttk.Label(frame, text="Enter Password:")
password_label.pack(pady=5)

password_frame = ttk.Frame(frame)
password_frame.pack()

password_entry = ttk.Entry(password_frame, show="*", width=30)
password_entry.pack(side="left", padx=5)

show_password_button = ttk.Button(password_frame, text="👁 Show", width=8, command=toggle_password)
show_password_button.pack(side="left")

# Encrypt Button
encrypt_button = ttk.Button(frame, text="🔐 Encrypt File", width=25, command=select_file_to_encrypt)
encrypt_button.pack(pady=10)

# Decrypt Button
decrypt_button = ttk.Button(frame, text="🔓 Decrypt File", width=25, command=select_file_to_decrypt)
decrypt_button.pack(pady=10)

# Add padding for a cleaner look
for widget in frame.winfo_children():
    widget.pack_configure(pady=5)

# Run the GUI
root.mainloop()
