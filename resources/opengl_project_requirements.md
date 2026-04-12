# OpenGL Project Requirements

This document summarizes the exact OpenGL components, mathematical formulas, and shader logic extracted from the lecture slides for implementation in the project.

---

## 1. OpenGL Components

### Functions & API
- **Environment Setup (GLFW/GLAD):**
  - `glfwInit()`: Initialize the GLFW library.
  - `glfwWindowHint()`: Set window configuration options (e.g., OpenGL version).
  - `glfwCreateWindow()`: Create a window and its associated context.
  - `glfwMakeContextCurrent()`: Make the context of a specified window current.
  - `gladLoadGLLoader()`: Load OpenGL function pointers.
  - `glfwWindowShouldClose()`, `glfwSwapBuffers()`, `glfwPollEvents()`: Main render loop management.
  - `glfwTerminate()`: Clean up resources.

- **Buffer & Vertex Management:**
  - `glGenVertexArrays()`, `glBindVertexArray()`: Manage Vertex Array Objects (VAO).
  - `glGenBuffers()`, `glBindBuffer()`, `glBufferData()`: Manage Vertex Buffer Objects (VBO) and Element Buffer Objects (EBO).
  - `glVertexAttribPointer()`, `glEnableVertexAttribArray()`: Link vertex data to shader attributes.

- **Shader Management:**
  - `glCreateShader()`, `glShaderSource()`, `glCompileShader()`: Create and compile individual shaders.
  - `glCreateProgram()`, `glAttachShader()`, `glLinkProgram()`: Link shaders into a usable program.
  - `glUseProgram()`: Activate a shader program for rendering.

- **Rendering & State:**
  - `glClearColor()`, `glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)`: Clear buffers.
  - `glDrawArrays()`, `glDrawElements()`: Execute draw calls.
  - `glEnable(GL_DEPTH_TEST)`: Enable depth testing for 3D rendering.
  - `glViewport()`: Set the rendering viewport dimensions.

- **Texture Management:**
  - `glGenTextures()`, `glBindTexture()`: Manage texture objects.
  - `glTexImage2D()`: Load image data into a texture.
  - `glTexParameteri()`: Set texture wrapping and filtering parameters.

### Constants & Primitives
- **Geometric Primitives:** `GL_POINTS`, `GL_LINES`, `GL_LINE_STRIP`, `GL_LINE_LOOP`, `GL_TRIANGLES`, `GL_TRIANGLE_STRIP`, `GL_TRIANGLE_FAN`.
- **Buffer Targets:** `GL_ARRAY_BUFFER`, `GL_ELEMENT_ARRAY_BUFFER`.
- **Shader Stages:** `GL_VERTEX_SHADER`, `GL_FRAGMENT_SHADER`.
- **Texture Parameters:** `GL_TEXTURE_2D`, `GL_TEXTURE_MAG_FILTER`, `GL_TEXTURE_MIN_FILTER`, `GL_LINEAR`, `GL_NEAREST`, `GL_REPEAT`, `GL_CLAMP_TO_EDGE`.

---

## 2. Mathematical Formulas

### Transformations
- **2D Translation:** 
  - $x' = x + t_x$
  - $y' = y + t_y$
- **2D Scaling:** 
  - $x' = x \cdot s_x$
  - $y' = y \cdot s_y$
- **3D Camera View Space (LookAt):**
  - $N = P_0 - P_{ref}$ (View plane normal / Forward vector)
  - $U = V \times N$ (Right vector, where $V$ is the 'up' hint)
  - $V' = N \times U$ (Recomputed 'up' vector)

### Curves & Surfaces
- **Bezier Curve (Bernstein Basis):**
  - $P(t) = \sum_{i=0}^n B_i \cdot J_{n,i}(t)$
  - $J_{n,i}(t) = \binom{n}{i} t^i (1-t)^{n-i}$
- **Surface of Revolution:**
  - $x = r \cdot \cos(\theta)$
  - $z = -r \cdot \sin(\theta)$
  - $y = f(r)$ (Height mapped to radius or specific function)

### Fractals
- **Mandelbrot Set:** $z_{n+1} = z_n^2 + c$
- **Fractal Dimension ($D$):** $D = \frac{\log(N)}{\log(1/s)}$, where $N$ is the number of self-similar pieces and $s$ is the scaling factor.

---

## 3. Illumination & Shading Logic

### Lighting Equations (Phong Model)
- **Global Illumination:** $I = I_{ambient} + I_{diffuse} + I_{specular}$
- **Ambient:** $I_a = k_a \cdot L_a$
- **Diffuse (Lambertian):** $I_d = k_d \cdot L_d \cdot \max(L \cdot N, 0)$
- **Specular:** $I_s = k_s \cdot L_s \cdot \max(R \cdot V, 0)^n$ (where $n$ is the shininess factor)

### Shading Techniques
- **Gouraud Shading:** Per-vertex lighting calculation; colors are interpolated across the fragment.
- **Phong Shading:** Per-fragment lighting calculation; normals are interpolated, and the lighting model is evaluated for every pixel.

---

## 4. Shader Logic (GLSL)

### Data Types & Qualifiers
- **Types:** `float`, `vec2`, `vec3`, `vec4`, `mat4`, `sampler2D`.
- **Qualifiers:**
  - `layout (location = 0) in`: Input vertex attributes.
  - `out`: Data passed to the next shader stage.
  - `uniform`: Global variables updated from CPU (e.g., transformation matrices, light properties).

### GLSL Logic Snippets
- **Standard Vertex Transformation:**
  ```glsl
  layout (location = 0) in vec3 aPos;
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 projection;
  void main() {
      gl_Position = projection * view * model * vec4(aPos, 1.0);
  }
  ```
- **Fragment Lighting (Diffuse Example):**
  ```glsl
  void main() {
      vec3 norm = normalize(Normal);
      vec3 lightDir = normalize(lightPos - FragPos);
      float diff = max(dot(norm, lightDir), 0.0);
      vec3 diffuse = diff * lightColor;
      FragColor = vec4(result, 1.0);
  }
  ```

---

## 5. Texture & Fractal Algorithms

- **Texture Filtering:** `GL_NEAREST` (blocky) vs `GL_LINEAR` (smooth).
- **Mipmaps:** Automatically generated levels of detail for distant textures.
- **Koch Snowflake:** Recursive line segment division.
- **Sierpinski Gasket:** Recursive triangle subdivision/removal.
