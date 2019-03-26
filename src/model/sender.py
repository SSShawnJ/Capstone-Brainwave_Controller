class BluetoothSender:
	def __init__(self, bluetooth, required_num):
		self.bluetooth = bluetooth
		self.count = 0
		self.current_signal = b'0'
		self.required_num = required_num

	def send(self, signal):
		if signal == self.current_signal:
			self.count += 1
		else:
			self.current_signal = signal
			self.count = 1

		if self.count == seld.required_num:
			# send control signal to arduino
			self.bluetooth.write(self.current_signal)
			# reset counter
			self.count == 0



		