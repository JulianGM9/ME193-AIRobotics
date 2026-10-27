import legoeducation as le

# Update these to match your motor's Connection Card
card_color = le.LEGO_COLOR_YELLOW
card_serial = '1131'

# Connect to the Single Motor
singlemotor = le.SingleMotor()
singlemotor.connect(card_color=card_color, card_serial=card_serial)

# Check connection
if not singlemotor.connected:
    print('Error connecting to Single Motor.')
    exit(1)

# Run the motor 180 degrees, counter-clockwise, at 100% speed
singlemotor.motor_run_for_degrees(
    720,
    direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE,
    speed=100
)

# Disconnect
singlemotor.disconnect()
exit(0)