#pragma once

#include <iostream>

#include "const.cuh"

#include "SDL.h"

int nx = RES_X;
int ny = RES_Y;

SDL_Window* window;
SDL_Renderer* renderer;

bool mouse_drag = false;

float theta = 80.0f * DEG2RAD;
float phi = 45.0f * DEG2RAD;

inline void setColor(float r, float g, float b, float a = 1.0) {
	r = (r > 1.0f) ? 1.0f : ((r < 0.0f) ? 0.0f : r);
	g = (g > 1.0f) ? 1.0f : ((g < 0.0f) ? 0.0f : g);
	b = (b > 1.0f) ? 1.0f : ((b < 0.0f) ? 0.0f : b);
	a = (a > 1.0f) ? 1.0f : ((a < 0.0f) ? 0.0f : a);
	SDL_SetRenderDrawColor(renderer, Uint8(r * 255.0), Uint8(g * 255.0), Uint8(b * 255.0), Uint8(a * 255.0));
}

inline void setPixel(int x, int y) {
	SDL_RenderDrawPoint(renderer, x, ny - y - 1);
}

inline void setPixel(int x, int y, float r, float g, float b, float a = 1.0) {
	setColor(r, g, b, a);
	setPixel(x, y);
}

inline void drawLine(int x1, int y1, int x2, int y2) {
	SDL_RenderDrawLine(renderer, x1, ny - y1 - 1, x2, ny - y2 - 1);
}

bool saveScreenshotBMP(std::string filepath, SDL_Window* SDLWindow = window, SDL_Renderer* SDLRenderer = renderer) {
	SDL_Surface* saveSurface = NULL;
	SDL_Surface* infoSurface = NULL;
	infoSurface = SDL_GetWindowSurface(SDLWindow);
	if (infoSurface == NULL) {
		std::cerr << "Failed to create info surface from window in saveScreenshotBMP(string), SDL_GetError() - " << SDL_GetError() << "\n";
	}
	else {
		unsigned char* pixels = new (std::nothrow) unsigned char[infoSurface->w * infoSurface->h * infoSurface->format->BytesPerPixel];
		if (pixels == 0) {
			std::cerr << "Unable to allocate memory for screenshot pixel data buffer!\n";
			return false;
		}
		else {
			if (SDL_RenderReadPixels(SDLRenderer, &infoSurface->clip_rect, infoSurface->format->format, pixels, infoSurface->w * infoSurface->format->BytesPerPixel) != 0) {
				std::cerr << "Failed to read pixel data from SDL_Renderer object. SDL_GetError() - " << SDL_GetError() << "\n";
				pixels = NULL;
				return false;
			}
			else {
				saveSurface = SDL_CreateRGBSurfaceFrom(pixels, infoSurface->w, infoSurface->h, infoSurface->format->BitsPerPixel, infoSurface->w * infoSurface->format->BytesPerPixel, infoSurface->format->Rmask, infoSurface->format->Gmask, infoSurface->format->Bmask, infoSurface->format->Amask);
				if (saveSurface == NULL) {
					std::cerr << "Couldn't create SDL_Surface from renderer pixel data. SDL_GetError() - " << SDL_GetError() << "\n";
					return false;
				}
				SDL_SaveBMP(saveSurface, filepath.c_str());
				SDL_FreeSurface(saveSurface);
				saveSurface = NULL;
			}
			delete[] pixels;
		}
		SDL_FreeSurface(infoSurface);
		infoSurface = NULL;
	}
	return true;
}

int init() {
	/* Initialize SDL2. */
	if (SDL_Init(SDL_INIT_VIDEO) != 0) {
		std::cout << "SDL_Init Error: " << SDL_GetError() << std::endl;
		return 1;
	}

	/* Create the window where we will draw. */
	window = SDL_CreateWindow("Ray Tracing with CUDA", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, nx, ny, SDL_WINDOW_SHOWN);
	if (window == nullptr) {
		std::cout << "SDL_CreateWindow Error: " << SDL_GetError() << std::endl;
		SDL_Quit();
		return 1;
	}

	/* We must call SDL_CreateRenderer in order for draw calls to affect this window. */
	renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	if (renderer == nullptr) {
		SDL_DestroyWindow(window);
		std::cout << "SDL_CreateRenderer Error: " << SDL_GetError() << std::endl;
		SDL_Quit();
		return 1;
	}
	return 0;
}

void close() {
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
}

void poll_event(SDL_Event& event) {
	while (SDL_PollEvent(&event)) {
		switch (event.type) {
		case SDL_KEYDOWN:
			switch (event.key.keysym.sym) {
			case SDLK_s:
				saveScreenshotBMP("SCREEN.bmp");
				break;
			}
		case SDL_MOUSEMOTION:
			if (mouse_drag) {
				int mx = event.motion.xrel;
				int my = event.motion.yrel;
				theta += -my * DEG2RAD;
				if (theta < DEG2RAD) {
					theta = DEG2RAD;
				}
				if (theta > (M_PI_2 - DEG2RAD)) {
					theta = M_PI_2 - DEG2RAD;
				}
				phi += -mx * DEG2RAD;
			}
			break;
		case SDL_MOUSEBUTTONDOWN:
			mouse_drag = true;
			break;
		case SDL_MOUSEBUTTONUP:
			mouse_drag = false;
			break;
		}
	}
}