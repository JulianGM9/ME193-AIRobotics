import legoeducation as le

# update these values to match the Connection Card
card_color = le.LEGO_COLOR_YELLOW
card_serial = '1131'

# Connect to the Double Motor
doublemotor = le.DoubleMotor()
doublemotor.connect(card_color=card_color, card_serial=card_serial)

# Check connection
if not doublemotor.connected:
	print('Error connecting to Double Motor.')
	exit(1) # error connecting

# Drive for 180-degrees
doublemotor.movement_move_for_degrees(1000)


# Disconnect
doublemotor.disconnect()
exit(0) # successful execution