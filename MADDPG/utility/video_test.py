import pygame
import numpy as np
import cv2

# Initialize Pygame
pygame.init()

# Set screen dimensions
width, height = 640, 480
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Create a VideoWriter object to save the frames as a video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (width, height))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Your rendering code here
    # For example:
    render_data = np.random.randint(0, 255, (width, height, 3), dtype=np.uint8)
    text_surface = pygame.Surface((100, 100))
    text_surface.fill((255, 255, 255))

    # Create a Pygame surface from render_data
    image = pygame.surfarray.make_surface(render_data)

    # Blit the surfaces onto the screen
    screen.blit(image, (0, 0))
    screen.blit(text_surface, (330, 200))
    pygame.display.update()

    # Capture the current frame
    frame = pygame.surfarray.array3d(screen)

    # Convert the Pygame surface to BGR format (OpenCV format)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Write the frame to the video file
    out.write(frame)

    clock.tick(30)  # Limit the frame rate to 30 FPS

out.release()  # Release the VideoWriter
pygame.quit()  # Quit Pygame

