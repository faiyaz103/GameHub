#include <cstddef> 
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "shader.h"
#include <iostream>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace std;
using namespace glm;

// --- Function Prototypes ---
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void processInput(GLFWwindow* window);
unsigned int loadTexture(char const* path);
void drawCube(unsigned int VAO, Shader& shader, glm::mat4 model, glm::vec3 color, unsigned int textureID = 0);
void drawLightCube(unsigned int VAO, Shader& shader, glm::mat4 model, glm::vec3 color);
void drawGameHubScene(unsigned int VAO, Shader& shader, unsigned int poolTexture);
void drawAllLights(unsigned int VAO, Shader& lightCubeShader, glm::mat4 projection, glm::mat4 view);
void setLightUniforms(Shader& shader);

// --- Settings ---
const unsigned int SCR_WIDTH = 1200;
const unsigned int SCR_HEIGHT = 800;

// --- Camera ---
glm::vec3 cameraPos   = glm::vec3(0.0f, 4.0f, 8.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, -0.3f, -1.0f);
glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f, 0.0f);
bool firstMouse = true;
float cameraYaw   = -90.0f;
float cameraPitch = -20.0f;
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// =========================================
// LIGHT SOURCE DEFINITIONS
// =========================================
// 3 Spotlights: above Pool Table, Table Tennis, Carrom Board
// 4 Point Lights: one on each wall (North, South, East, West) at upper-middle
// No directional light - scene is lit only by physical bulbs

// Spotlight positions (hanging from ceiling at y=5.7, above each board center)
const glm::vec3 SPOT_POSITIONS[3] = {
    glm::vec3(-3.0f, 5.7f,  0.0f),  // Pool Table
    glm::vec3( 4.0f, 5.7f, -2.0f),  // Table Tennis
    glm::vec3( 4.0f, 5.7f,  3.0f)   // Carrom Board
};

// Point light positions: mounted on wall surfaces, upper-middle of each wall
// Room walls: x=±8, z=±6, roof at y=6
// Bulbs sit at y=5.0 (upper area), slightly inside the wall
const glm::vec3 POINT_POSITIONS[4] = {
    glm::vec3( 0.0f, 5.0f, -5.85f),  // North wall (z=-6) facing inward
    glm::vec3( 0.0f, 5.0f,  5.85f),  // South wall (z=+6) facing inward
    glm::vec3( 7.85f, 5.0f,  0.0f),  // East wall  (x=+8) facing inward
    glm::vec3(-7.85f, 5.0f,  0.0f)   // West wall  (x=-8) facing inward
};

// Toggle states
bool spotLightOn[3]  = { true, true, true  };
bool pointLightOn[4] = { true, true, true, true };
bool ambientOn   = true;
bool diffuseOn   = true;
bool specularOn  = true;
bool isSplitView = false;
bool keyProcessed[1024] = { false };

// Shading & Texture
bool useGouraud      = false; // Default: Phong (per-fragment)
int  globalTextureMode = 1;   // 0=None, 1=Texture, 2=Blend

// ==========================================
// 1. PHONG VERTEX SHADER
// ==========================================
const std::string vertexShaderPhongSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in vec2 aTexCoords;

    out vec3 FragPos;
    out vec3 Normal;
    out vec2 TexCoords;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        FragPos   = vec3(model * vec4(aPos, 1.0));
        Normal    = mat3(transpose(inverse(model))) * aNormal;
        TexCoords = aTexCoords;
        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
)";

// ==========================================
// 2. PHONG FRAGMENT SHADER (Per-Fragment)
// Implements full Phong model:
//   I = I_ambient + I_diffuse + I_specular
//   I_a = k_a * L_a
//   I_d = k_d * L_d * max(L . N, 0)
//   I_s = k_s * L_s * max(R . V, 0)^n
// ==========================================
const std::string fragmentShaderPhongSource = R"(
    #version 330 core
    out vec4 FragColor;

    #define NR_POINT_LIGHTS 4
    #define NR_SPOT_LIGHTS  3

    struct PointLight {
        vec3  position;
        float constant;
        float linear;
        float quadratic;
        vec3  ambient;
        vec3  diffuse;
        vec3  specular;
    };

    struct SpotLight {
        vec3  position;
        vec3  direction;
        float cutOff;
        float outerCutOff;
        float constant;
        float linear;
        float quadratic;
        vec3  ambient;
        vec3  diffuse;
        vec3  specular;
    };

    in vec3 FragPos;
    in vec3 Normal;
    in vec2 TexCoords;

    uniform vec3 viewPos;
    uniform vec3 objectColor;
    uniform sampler2D diffuseMap;
    uniform int  textureMode;  // 0=None, 1=Texture, 2=Blend

    uniform PointLight pointLights[NR_POINT_LIGHTS];
    uniform SpotLight  spotLights[NR_SPOT_LIGHTS];
    uniform bool enablePoint[NR_POINT_LIGHTS];
    uniform bool enableSpot[NR_SPOT_LIGHTS];
    uniform bool enableAmbient;
    uniform bool enableDiffuse;
    uniform bool enableSpecular;

    // --- Point Light Calculation ---
    vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
        vec3  lightDir   = normalize(light.position - fragPos);
        float diff       = max(dot(normal, lightDir), 0.0);
        vec3  reflectDir = reflect(-lightDir, normal);
        float spec       = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        float dist       = length(light.position - fragPos);
        float atten      = 1.0 / (light.constant + light.linear * dist + light.quadratic * dist * dist);

        vec3 ambient  = enableAmbient  ? light.ambient  * atten       : vec3(0.0);
        vec3 diffuse  = enableDiffuse  ? light.diffuse  * diff * atten : vec3(0.0);
        vec3 specular = enableSpecular ? light.specular * spec * atten : vec3(0.0);
        return ambient + diffuse + specular;
    }

    // --- Spotlight Calculation (Phong + soft edge) ---
    vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
        vec3  lightDir = normalize(light.position - fragPos);
        float theta    = dot(lightDir, normalize(-light.direction));
        float epsilon  = light.cutOff - light.outerCutOff;
        float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);

        float diff       = max(dot(normal, lightDir), 0.0);
        vec3  reflectDir = reflect(-lightDir, normal);
        float spec       = pow(max(dot(viewDir, reflectDir), 0.0), 64.0);
        float dist       = length(light.position - fragPos);
        float atten      = 1.0 / (light.constant + light.linear * dist + light.quadratic * dist * dist);

        vec3 ambient  = enableAmbient  ? light.ambient  * atten                  : vec3(0.0);
        vec3 diffuse  = enableDiffuse  ? light.diffuse  * diff * atten * intensity : vec3(0.0);
        vec3 specular = enableSpecular ? light.specular * spec * atten * intensity : vec3(0.0);
        return ambient + diffuse + specular;
    }

    void main() {
        vec3 norm    = normalize(Normal);
        vec3 viewDir = normalize(viewPos - FragPos);

        // Small global ambient fill so darkened areas aren't pitch black
        vec3 result = vec3(0.03);

        for (int i = 0; i < NR_POINT_LIGHTS; i++)
            if (enablePoint[i]) result += CalcPointLight(pointLights[i], norm, FragPos, viewDir);
        for (int i = 0; i < NR_SPOT_LIGHTS; i++)
            if (enableSpot[i])  result += CalcSpotLight(spotLights[i],  norm, FragPos, viewDir);

        vec4 texColor;
        if      (textureMode == 0) texColor = vec4(objectColor, 1.0);
        else if (textureMode == 1) texColor = texture(diffuseMap, TexCoords);
        else                       texColor = mix(vec4(objectColor, 1.0), texture(diffuseMap, TexCoords), 0.5);

        FragColor = vec4(result * texColor.rgb, texColor.a);
    }
)";

// ==========================================
// 3. GOURAUD VERTEX SHADER (Per-Vertex)
// Lighting computed at each vertex; color
// interpolated across the fragment.
// ==========================================
const std::string vertexShaderGouraudSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in vec2 aTexCoords;

    out vec3 LightingColor;
    out vec2 TexCoords;

    #define NR_POINT_LIGHTS 4
    #define NR_SPOT_LIGHTS  3

    struct PointLight {
        vec3  position;
        float constant;
        float linear;
        float quadratic;
        vec3  ambient;
        vec3  diffuse;
        vec3  specular;
    };

    struct SpotLight {
        vec3  position;
        vec3  direction;
        float cutOff;
        float outerCutOff;
        float constant;
        float linear;
        float quadratic;
        vec3  ambient;
        vec3  diffuse;
        vec3  specular;
    };

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform vec3 viewPos;

    uniform PointLight pointLights[NR_POINT_LIGHTS];
    uniform SpotLight  spotLights[NR_SPOT_LIGHTS];
    uniform bool enablePoint[NR_POINT_LIGHTS];
    uniform bool enableSpot[NR_SPOT_LIGHTS];
    uniform bool enableAmbient;
    uniform bool enableDiffuse;
    uniform bool enableSpecular;

    vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
        vec3  lightDir   = normalize(light.position - fragPos);
        float diff       = max(dot(normal, lightDir), 0.0);
        vec3  reflectDir = reflect(-lightDir, normal);
        float spec       = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        float dist       = length(light.position - fragPos);
        float atten      = 1.0 / (light.constant + light.linear * dist + light.quadratic * dist * dist);
        vec3 ambient  = enableAmbient  ? light.ambient  * atten        : vec3(0.0);
        vec3 diffuse  = enableDiffuse  ? light.diffuse  * diff * atten  : vec3(0.0);
        vec3 specular = enableSpecular ? light.specular * spec * atten  : vec3(0.0);
        return ambient + diffuse + specular;
    }

    vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
        vec3  lightDir  = normalize(light.position - fragPos);
        float theta     = dot(lightDir, normalize(-light.direction));
        float epsilon   = light.cutOff - light.outerCutOff;
        float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
        float diff      = max(dot(normal, lightDir), 0.0);
        vec3  reflectDir = reflect(-lightDir, normal);
        float spec       = pow(max(dot(viewDir, reflectDir), 0.0), 64.0);
        float dist       = length(light.position - fragPos);
        float atten      = 1.0 / (light.constant + light.linear * dist + light.quadratic * dist * dist);
        vec3 ambient  = enableAmbient  ? light.ambient  * atten                   : vec3(0.0);
        vec3 diffuse  = enableDiffuse  ? light.diffuse  * diff * atten * intensity  : vec3(0.0);
        vec3 specular = enableSpecular ? light.specular * spec * atten * intensity  : vec3(0.0);
        return ambient + diffuse + specular;
    }

    void main() {
        vec3 FragPos = vec3(model * vec4(aPos, 1.0));
        vec3 Normal  = normalize(mat3(transpose(inverse(model))) * aNormal);
        TexCoords    = aTexCoords;
        vec3 viewDir = normalize(viewPos - FragPos);

        vec3 result = vec3(0.03); // base ambient fill
        for (int i = 0; i < NR_POINT_LIGHTS; i++)
            if (enablePoint[i]) result += CalcPointLight(pointLights[i], Normal, FragPos, viewDir);
        for (int i = 0; i < NR_SPOT_LIGHTS; i++)
            if (enableSpot[i])  result += CalcSpotLight(spotLights[i],  Normal, FragPos, viewDir);

        LightingColor = result;
        gl_Position   = projection * view * vec4(FragPos, 1.0);
    }
)";

// ==========================================
// 4. GOURAUD FRAGMENT SHADER
// ==========================================
const std::string fragmentShaderGouraudSource = R"(
    #version 330 core
    out vec4 FragColor;

    in vec3 LightingColor;
    in vec2 TexCoords;

    uniform vec3 objectColor;
    uniform sampler2D diffuseMap;
    uniform int  textureMode;

    void main() {
        vec4 texColor;
        if      (textureMode == 0) texColor = vec4(objectColor, 1.0);
        else if (textureMode == 1) texColor = texture(diffuseMap, TexCoords);
        else                       texColor = mix(vec4(objectColor, 1.0), texture(diffuseMap, TexCoords), 0.5);

        FragColor = vec4(LightingColor * texColor.rgb, texColor.a);
    }
)";

// ==========================================
// 5. LIGHT CUBE SHADER (Unlit emissive bulb)
// ==========================================
const std::string vertexShaderLightCubeSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    void main() {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
    }
)";

const std::string fragmentShaderLightCubeSource = R"(
    #version 330 core
    out vec4 FragColor;
    uniform vec3 objectColor;
    void main() {
        // Unlit: no lighting calculation, appears fully emissive
        FragColor = vec4(objectColor, 1.0);
    }
)";

// ==========================================
// MAIN
// ==========================================
int main()
{
    // --- GLFW Init & Window ---
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "GameHub", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    // --- Compile Shaders ---
    Shader phongShader(vertexShaderPhongSource,    fragmentShaderPhongSource);
    Shader gouraudShader(vertexShaderGouraudSource, fragmentShaderGouraudSource);
    Shader lightCubeShader(vertexShaderLightCubeSource, fragmentShaderLightCubeSource);

    // --- Textures ---
    unsigned int poolStickerTexture = loadTexture("resources/Designer.png");

    // --- Cube Vertex Data: positions, normals, texCoords ---
    float vertices[] = {
        // positions          // normals           // tex coords
        -0.5f,-0.5f,-0.5f,  0.0f, 0.0f,-1.0f,  0.0f,0.0f,
         0.5f,-0.5f,-0.5f,  0.0f, 0.0f,-1.0f,  1.0f,0.0f,
         0.5f, 0.5f,-0.5f,  0.0f, 0.0f,-1.0f,  1.0f,1.0f,
         0.5f, 0.5f,-0.5f,  0.0f, 0.0f,-1.0f,  1.0f,1.0f,
        -0.5f, 0.5f,-0.5f,  0.0f, 0.0f,-1.0f,  0.0f,1.0f,
        -0.5f,-0.5f,-0.5f,  0.0f, 0.0f,-1.0f,  0.0f,0.0f,

        -0.5f,-0.5f, 0.5f,  0.0f, 0.0f, 1.0f,  0.0f,0.0f,
         0.5f,-0.5f, 0.5f,  0.0f, 0.0f, 1.0f,  1.0f,0.0f,
         0.5f, 0.5f, 0.5f,  0.0f, 0.0f, 1.0f,  1.0f,1.0f,
         0.5f, 0.5f, 0.5f,  0.0f, 0.0f, 1.0f,  1.0f,1.0f,
        -0.5f, 0.5f, 0.5f,  0.0f, 0.0f, 1.0f,  0.0f,1.0f,
        -0.5f,-0.5f, 0.5f,  0.0f, 0.0f, 1.0f,  0.0f,0.0f,

        -0.5f, 0.5f, 0.5f, -1.0f, 0.0f, 0.0f,  1.0f,0.0f,
        -0.5f, 0.5f,-0.5f, -1.0f, 0.0f, 0.0f,  1.0f,1.0f,
        -0.5f,-0.5f,-0.5f, -1.0f, 0.0f, 0.0f,  0.0f,1.0f,
        -0.5f,-0.5f,-0.5f, -1.0f, 0.0f, 0.0f,  0.0f,1.0f,
        -0.5f,-0.5f, 0.5f, -1.0f, 0.0f, 0.0f,  0.0f,0.0f,
        -0.5f, 0.5f, 0.5f, -1.0f, 0.0f, 0.0f,  1.0f,0.0f,

         0.5f, 0.5f, 0.5f,  1.0f, 0.0f, 0.0f,  1.0f,0.0f,
         0.5f, 0.5f,-0.5f,  1.0f, 0.0f, 0.0f,  1.0f,1.0f,
         0.5f,-0.5f,-0.5f,  1.0f, 0.0f, 0.0f,  0.0f,1.0f,
         0.5f,-0.5f,-0.5f,  1.0f, 0.0f, 0.0f,  0.0f,1.0f,
         0.5f,-0.5f, 0.5f,  1.0f, 0.0f, 0.0f,  0.0f,0.0f,
         0.5f, 0.5f, 0.5f,  1.0f, 0.0f, 0.0f,  1.0f,0.0f,

        -0.5f,-0.5f,-0.5f,  0.0f,-1.0f, 0.0f,  0.0f,1.0f,
         0.5f,-0.5f,-0.5f,  0.0f,-1.0f, 0.0f,  1.0f,1.0f,
         0.5f,-0.5f, 0.5f,  0.0f,-1.0f, 0.0f,  1.0f,0.0f,
         0.5f,-0.5f, 0.5f,  0.0f,-1.0f, 0.0f,  1.0f,0.0f,
        -0.5f,-0.5f, 0.5f,  0.0f,-1.0f, 0.0f,  0.0f,0.0f,
        -0.5f,-0.5f,-0.5f,  0.0f,-1.0f, 0.0f,  0.0f,1.0f,

        -0.5f, 0.5f,-0.5f,  0.0f, 1.0f, 0.0f,  0.0f,1.0f,
         0.5f, 0.5f,-0.5f,  0.0f, 1.0f, 0.0f,  1.0f,1.0f,
         0.5f, 0.5f, 0.5f,  0.0f, 1.0f, 0.0f,  1.0f,0.0f,
         0.5f, 0.5f, 0.5f,  0.0f, 1.0f, 0.0f,  1.0f,0.0f,
        -0.5f, 0.5f, 0.5f,  0.0f, 1.0f, 0.0f,  0.0f,0.0f,
        -0.5f, 0.5f,-0.5f,  0.0f, 1.0f, 0.0f,  0.0f,1.0f
    };

    // --- VAO / VBO ---
    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // normal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texcoord
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // ==========================================
    // RENDER LOOP
    // ==========================================
    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        glClearColor(0.04f, 0.04f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        Shader& activeShader = useGouraud ? gouraudShader : phongShader;
        activeShader.use();
        setLightUniforms(activeShader);
        activeShader.setVec3("viewPos", cameraPos);

        int displayW, displayH;
        glfwGetFramebufferSize(window, &displayW, &displayH);

        if (!isSplitView) {
            glViewport(0, 0, displayW, displayH);
            glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)displayW / displayH, 0.1f, 100.0f);
            glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
            activeShader.setMat4("projection", proj);
            activeShader.setMat4("view", view);
            drawGameHubScene(VAO, activeShader, poolStickerTexture);
            drawAllLights(VAO, lightCubeShader, proj, view);
        }
        else {
            float ar = (float)(displayW / 2) / (float)(displayH / 2);
            glm::mat4 proj = glm::perspective(glm::radians(45.0f), ar, 0.1f, 100.0f);
            glm::mat4 view;

            // Top-down
            glViewport(0, displayH / 2, displayW / 2, displayH / 2);
            view = glm::lookAt(glm::vec3(0.0f, 15.0f, 0.1f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            activeShader.use(); activeShader.setMat4("projection", proj); activeShader.setMat4("view", view);
            drawGameHubScene(VAO, activeShader, poolStickerTexture);
            drawAllLights(VAO, lightCubeShader, proj, view);

            // Right side
            glViewport(displayW / 2, displayH / 2, displayW / 2, displayH / 2);
            view = glm::lookAt(glm::vec3(14.0f, 3.0f, 0.0f), glm::vec3(0.0f, 3.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            activeShader.use(); activeShader.setMat4("view", view);
            drawGameHubScene(VAO, activeShader, poolStickerTexture);
            drawAllLights(VAO, lightCubeShader, proj, view);

            // Front
            glViewport(0, 0, displayW / 2, displayH / 2);
            view = glm::lookAt(glm::vec3(0.0f, 2.0f, 14.0f), glm::vec3(0.0f, 2.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            activeShader.use(); activeShader.setMat4("view", view);
            drawGameHubScene(VAO, activeShader, poolStickerTexture);
            drawAllLights(VAO, lightCubeShader, proj, view);

            // Free camera
            glViewport(displayW / 2, 0, displayW / 2, displayH / 2);
            view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
            activeShader.use(); activeShader.setMat4("view", view);
            drawGameHubScene(VAO, activeShader, poolStickerTexture);
            drawAllLights(VAO, lightCubeShader, proj, view);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glfwTerminate();
    return 0;
}

// ==========================================
// SET LIGHT UNIFORMS
// ==========================================
void setLightUniforms(Shader& shader) {
    shader.setBool("enableAmbient",  ambientOn);
    shader.setBool("enableDiffuse",  diffuseOn);
    shader.setBool("enableSpecular", specularOn);

    // --- 3 Spotlights above each board ---
    // Warm bulb color (slightly yellow-white like an incandescent)
    glm::vec3 bulbDiffuse  = glm::vec3(1.0f, 0.92f, 0.75f);
    glm::vec3 bulbSpecular = glm::vec3(1.0f, 0.95f, 0.85f);

    for (int i = 0; i < 3; i++) {
        std::string n = "spotLights[" + std::to_string(i) + "].";
        shader.setBool("enableSpot[" + std::to_string(i) + "]", spotLightOn[i]);
        shader.setVec3(n + "position",   SPOT_POSITIONS[i]);
        shader.setVec3(n + "direction",  glm::vec3(0.0f, -1.0f, 0.0f)); // aimed straight down
        shader.setVec3(n + "ambient",    glm::vec3(0.0f));               // spotlights have no ambient
        shader.setVec3(n + "diffuse",    bulbDiffuse);
        shader.setVec3(n + "specular",   bulbSpecular);
        shader.setFloat(n + "constant",  1.0f);
        shader.setFloat(n + "linear",    0.07f);
        shader.setFloat(n + "quadratic", 0.017f);
        shader.setFloat(n + "cutOff",      glm::cos(glm::radians(18.0f)));
        shader.setFloat(n + "outerCutOff", glm::cos(glm::radians(25.0f)));
    }

    // --- 4 Wall Point Lights ---
    // Cool-white wall lights (like fluorescent brackets)
    glm::vec3 wallDiffuse  = glm::vec3(0.75f, 0.80f, 0.85f);
    glm::vec3 wallSpecular = glm::vec3(0.50f, 0.55f, 0.60f);

    for (int i = 0; i < 4; i++) {
        std::string n = "pointLights[" + std::to_string(i) + "].";
        shader.setBool("enablePoint[" + std::to_string(i) + "]", pointLightOn[i]);
        shader.setVec3(n + "position",  POINT_POSITIONS[i]);
        shader.setVec3(n + "ambient",   glm::vec3(0.05f)); // very slight ambient glow
        shader.setVec3(n + "diffuse",   wallDiffuse);
        shader.setVec3(n + "specular",  wallSpecular);
        shader.setFloat(n + "constant",  1.0f);
        shader.setFloat(n + "linear",    0.045f);
        shader.setFloat(n + "quadratic", 0.0075f);
    }
}

// ==========================================
// LOAD TEXTURE
// ==========================================
unsigned int loadTexture(char const* path) {
    unsigned int textureID;
    glGenTextures(1, &textureID);
    int width, height, nrComponents;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(path, &width, &height, &nrComponents, 0);
    if (data) {
        GLenum format = GL_RGB;
        if      (nrComponents == 1) format = GL_RED;
        else if (nrComponents == 3) format = GL_RGB;
        else if (nrComponents == 4) format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        stbi_image_free(data);
    } else {
        std::cout << "Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
    }
    return textureID;
}

// ==========================================
// DRAW HELPERS
// ==========================================
void drawCube(unsigned int VAO, Shader& shader, glm::mat4 model, glm::vec3 color, unsigned int textureID) {
    shader.setMat4("model", model);
    shader.setVec3("objectColor", color);
    if (textureID > 0) {
        shader.setInt("textureMode", globalTextureMode);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);
        shader.setInt("diffuseMap", 0);
    } else {
        shader.setInt("textureMode", 0);
    }
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
}

void drawLightCube(unsigned int VAO, Shader& shader, glm::mat4 model, glm::vec3 color) {
    shader.setMat4("model", model);
    shader.setVec3("objectColor", color);
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
}

// ==========================================
// DRAW ALL LIGHT FIXTURES
// Spotlights: bulb + thin cord from ceiling
// Wall lights: flat bracket flush against wall
// ==========================================
void drawAllLights(unsigned int VAO, Shader& lightCubeShader, glm::mat4 projection, glm::mat4 view) {
    lightCubeShader.use();
    lightCubeShader.setMat4("projection", projection);
    lightCubeShader.setMat4("view", view);

    glm::vec3 bulbColor   = glm::vec3(1.0f, 0.95f, 0.75f); // warm incandescent
    glm::vec3 cordColor   = glm::vec3(0.15f, 0.12f, 0.10f); // dark cord
    glm::vec3 bracketColor = glm::vec3(0.8f, 0.8f, 0.8f);  // metal bracket

    // --- 3 SPOTLIGHTS: hanging from ceiling ---
    for (int i = 0; i < 3; i++) {
        if (!spotLightOn[i]) continue;

        glm::vec3 pos = SPOT_POSITIONS[i];

        // Cord: thin vertical line from ceiling (y=6.0) to bulb (y=pos.y+0.12)
        float cordLen  = 6.0f - (pos.y + 0.13f);
        float cordMid  = 6.0f - cordLen / 2.0f;
        glm::mat4 cordModel = glm::scale(
            glm::translate(glm::mat4(1.0f), glm::vec3(pos.x, cordMid, pos.z)),
            glm::vec3(0.025f, cordLen, 0.025f)
        );
        drawLightCube(VAO, lightCubeShader, cordModel, cordColor);

        // Bulb socket: small dark cube just above the bulb
        glm::mat4 socketModel = glm::scale(
            glm::translate(glm::mat4(1.0f), glm::vec3(pos.x, pos.y + 0.13f, pos.z)),
            glm::vec3(0.12f, 0.10f, 0.12f)
        );
        drawLightCube(VAO, lightCubeShader, socketModel, glm::vec3(0.2f, 0.2f, 0.2f));

        // Bulb: bright glowing cube
        glm::mat4 bulbModel = glm::scale(
            glm::translate(glm::mat4(1.0f), pos),
            glm::vec3(0.18f, 0.18f, 0.18f)
        );
        drawLightCube(VAO, lightCubeShader, bulbModel, bulbColor);
    }

    // --- 4 WALL POINT LIGHTS: bracket mounted flush on wall ---
    for (int i = 0; i < 4; i++) {
        if (!pointLightOn[i]) continue;

        glm::vec3 pos = POINT_POSITIONS[i];

        // Bracket backing (flat, flush against wall)
        // We need to orient brackets depending on which wall
        glm::mat4 backingModel;
        glm::mat4 bulbModel;

        if (i == 0) {
            // North wall: z = -6, bracket faces +z
            backingModel = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(pos.x, pos.y, -5.97f)), glm::vec3(0.35f, 0.35f, 0.05f));
            bulbModel    = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(pos.x, pos.y, -5.78f)), glm::vec3(0.15f, 0.15f, 0.20f));
        } else if (i == 1) {
            // South wall: z = +6, bracket faces -z
            backingModel = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(pos.x, pos.y, 5.97f)), glm::vec3(0.35f, 0.35f, 0.05f));
            bulbModel    = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(pos.x, pos.y, 5.78f)), glm::vec3(0.15f, 0.15f, 0.20f));
        } else if (i == 2) {
            // East wall: x = +8, bracket faces -x
            backingModel = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(7.97f, pos.y, pos.z)), glm::vec3(0.05f, 0.35f, 0.35f));
            bulbModel    = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(7.78f, pos.y, pos.z)), glm::vec3(0.20f, 0.15f, 0.15f));
        } else {
            // West wall: x = -8, bracket faces +x
            backingModel = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-7.97f, pos.y, pos.z)), glm::vec3(0.05f, 0.35f, 0.35f));
            bulbModel    = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-7.78f, pos.y, pos.z)), glm::vec3(0.20f, 0.15f, 0.15f));
        }

        drawLightCube(VAO, lightCubeShader, backingModel, bracketColor);
        drawLightCube(VAO, lightCubeShader, bulbModel,    bulbColor);
    }
}

// ==========================================
// SCENE DRAWING
// ==========================================
void drawGameHubScene(unsigned int VAO, Shader& shader, unsigned int poolTexture) {
    glm::vec3 wallColor  = glm::vec3(0.85f, 0.88f, 0.90f);
    glm::vec3 floorColor = glm::vec3(0.38f, 0.38f, 0.38f);
    glm::vec3 roofColor  = glm::vec3(0.92f, 0.92f, 0.92f);
    glm::vec3 glassColor = glm::vec3(0.40f, 0.70f, 0.90f);

    // 1. ROOM
    // Floor
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f)), glm::vec3(16.0f, 0.1f, 12.0f)), floorColor);
    // Ceiling
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 6.0f, 0.0f)), glm::vec3(16.0f, 0.1f, 12.0f)), roofColor);
    // Back wall (North, z=-6)
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 3.0f, -6.0f)), glm::vec3(16.0f, 6.0f, 0.1f)), wallColor);

    // West wall (x=-8) with window
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-8.0f, 1.0f, 0.0f)),  glm::vec3(0.1f, 2.0f, 12.0f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-8.0f, 5.0f, 0.0f)),  glm::vec3(0.1f, 2.0f, 12.0f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-8.0f, 3.0f, -4.0f)), glm::vec3(0.1f, 2.0f, 4.0f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-8.0f, 3.0f,  4.0f)), glm::vec3(0.1f, 2.0f, 4.0f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-8.0f, 3.0f,  0.0f)), glm::vec3(0.05f, 2.0f, 4.0f)), glassColor);

    // East wall (x=+8) with window
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(8.0f, 1.0f, 0.0f)),   glm::vec3(0.1f, 2.0f, 12.0f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(8.0f, 5.0f, 0.0f)),   glm::vec3(0.1f, 2.0f, 12.0f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(8.0f, 3.0f, -4.0f)),  glm::vec3(0.1f, 2.0f, 4.0f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(8.0f, 3.0f,  4.0f)),  glm::vec3(0.1f, 2.0f, 4.0f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(8.0f, 3.0f,  0.0f)),  glm::vec3(0.05f, 2.0f, 4.0f)), glassColor);

    // Front wall (South, z=+6) with door
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-4.5f, 3.0f, 6.0f)), glm::vec3(7.0f, 6.0f, 0.1f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3( 4.5f, 3.0f, 6.0f)), glm::vec3(7.0f, 6.0f, 0.1f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3( 0.0f, 4.5f, 6.0f)), glm::vec3(2.0f, 3.0f, 0.1f)), wallColor);
    glm::mat4 door = glm::translate(glm::mat4(1.0f), glm::vec3(-1.0f, 1.5f, 6.0f));
    door = glm::rotate(door, glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    door = glm::translate(door, glm::vec3(1.0f, 0.0f, 0.0f));
    drawCube(VAO, shader, glm::scale(door, glm::vec3(2.0f, 3.0f, 0.05f)), glm::vec3(0.35f, 0.18f, 0.08f));

    // 2. POOL (BILLIARDS) TABLE
    glm::vec3 poolCenter = glm::vec3(-3.0f, 0.8f, 0.0f);
    glm::vec3 woodColor  = glm::vec3(0.30f, 0.15f, 0.05f);
    glm::vec3 feltGreen  = glm::vec3(0.08f, 0.45f, 0.18f);

    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), poolCenter), glm::vec3(3.0f, 0.1f, 4.5f)), feltGreen);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), poolCenter + glm::vec3( 0.0f, 0.1f, -2.3f)), glm::vec3(3.2f, 0.2f, 0.2f)), woodColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), poolCenter + glm::vec3( 0.0f, 0.1f,  2.3f)), glm::vec3(3.2f, 0.2f, 0.2f)), woodColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), poolCenter + glm::vec3(-1.6f, 0.1f,  0.0f)), glm::vec3(0.2f, 0.2f, 4.4f)), woodColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), poolCenter + glm::vec3( 1.6f, 0.1f,  0.0f)), glm::vec3(0.2f, 0.2f, 4.4f)), woodColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), poolCenter + glm::vec3(-1.3f,-0.4f, -2.0f)), glm::vec3(0.2f, 0.8f, 0.2f)), woodColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), poolCenter + glm::vec3( 1.3f,-0.4f, -2.0f)), glm::vec3(0.2f, 0.8f, 0.2f)), woodColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), poolCenter + glm::vec3(-1.3f,-0.4f,  2.0f)), glm::vec3(0.2f, 0.8f, 0.2f)), woodColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), poolCenter + glm::vec3( 1.3f,-0.4f,  2.0f)), glm::vec3(0.2f, 0.8f, 0.2f)), woodColor);
    // Cue stick
    glm::mat4 cue = glm::rotate(glm::translate(glm::mat4(1.0f), poolCenter + glm::vec3(0.0f, 0.4f, 1.2f)), glm::radians(-10.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    drawCube(VAO, shader, glm::scale(cue, glm::vec3(0.04f, 0.04f, 2.5f)), glm::vec3(0.80f, 0.70f, 0.50f));

    // 3. TABLE TENNIS
    glm::vec3 ttCenter  = glm::vec3(4.0f, 0.8f, -2.0f);
    glm::vec3 blueTable = glm::vec3(0.10f, 0.30f, 0.70f);
    glm::vec3 metalLegs = glm::vec3(0.20f, 0.20f, 0.20f);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), ttCenter), glm::vec3(3.0f, 0.05f, 1.6f)), blueTable);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), ttCenter + glm::vec3(0.0f, 0.10f, 0.0f)), glm::vec3(0.02f, 0.15f, 1.7f)), glm::vec3(0.9f));
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), ttCenter + glm::vec3(-1.4f,-0.4f,-0.7f)), glm::vec3(0.05f, 0.8f, 0.05f)), metalLegs);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), ttCenter + glm::vec3(-1.4f,-0.4f, 0.7f)), glm::vec3(0.05f, 0.8f, 0.05f)), metalLegs);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), ttCenter + glm::vec3( 1.4f,-0.4f,-0.7f)), glm::vec3(0.05f, 0.8f, 0.05f)), metalLegs);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), ttCenter + glm::vec3( 1.4f,-0.4f, 0.7f)), glm::vec3(0.05f, 0.8f, 0.05f)), metalLegs);

    // 4. CARROM BOARD
    glm::vec3 carromCenter = glm::vec3(4.0f, 0.6f, 3.0f);
    glm::vec3 frameWood    = glm::vec3(0.30f, 0.15f, 0.05f);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), carromCenter - glm::vec3(0.0f, 0.3f, 0.0f)), glm::vec3(0.8f, 0.6f, 0.8f)), metalLegs);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), carromCenter), glm::vec3(1.2f, 0.05f, 1.2f)), glm::vec3(0.80f, 0.60f, 0.40f));
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), carromCenter + glm::vec3( 0.0f, 0.05f,-0.65f)), glm::vec3(1.4f, 0.1f, 0.1f)), frameWood);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), carromCenter + glm::vec3( 0.0f, 0.05f, 0.65f)), glm::vec3(1.4f, 0.1f, 0.1f)), frameWood);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), carromCenter + glm::vec3(-0.65f, 0.05f, 0.0f)), glm::vec3(0.1f, 0.1f, 1.4f)), frameWood);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), carromCenter + glm::vec3( 0.65f, 0.05f, 0.0f)), glm::vec3(0.1f, 0.1f, 1.4f)), frameWood);
}

// ==========================================
// INPUT CONTROLS
// ==========================================
void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    float speed = 4.5f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) cameraPos += speed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) cameraPos -= speed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * speed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * speed;

    // Shading
    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_G]) { useGouraud = true;  cout << "Gouraud Shading\n"; keyProcessed[GLFW_KEY_G] = true; }
    } else keyProcessed[GLFW_KEY_G] = false;
    if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_H]) { useGouraud = false; cout << "Phong Shading\n";   keyProcessed[GLFW_KEY_H] = true; }
    } else keyProcessed[GLFW_KEY_H] = false;

    // Texture
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_Z]) { globalTextureMode = 0; cout << "Texture OFF\n";    keyProcessed[GLFW_KEY_Z] = true; }
    } else keyProcessed[GLFW_KEY_Z] = false;
    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_X]) { globalTextureMode = 1; cout << "Texture ON\n";     keyProcessed[GLFW_KEY_X] = true; }
    } else keyProcessed[GLFW_KEY_X] = false;
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_C]) { globalTextureMode = 2; cout << "Texture BLEND\n";  keyProcessed[GLFW_KEY_C] = true; }
    } else keyProcessed[GLFW_KEY_C] = false;

    // Light toggles  -  Number row keys
    // 1-3: Spotlights (above Pool, TT, Carrom)
    for (int i = 0; i < 3; i++) {
        int key = GLFW_KEY_1 + i;
        if (glfwGetKey(window, key) == GLFW_PRESS) {
            if (!keyProcessed[key]) {
                spotLightOn[i] = !spotLightOn[i];
                cout << "Spotlight[" << i << "] " << (spotLightOn[i] ? "ON" : "OFF") << "\n";
                keyProcessed[key] = true;
            }
        } else keyProcessed[key] = false;
    }
    // 4-7: Wall point lights (N, S, E, W)
    for (int i = 0; i < 4; i++) {
        int key = GLFW_KEY_4 + i;
        if (glfwGetKey(window, key) == GLFW_PRESS) {
            if (!keyProcessed[key]) {
                pointLightOn[i] = !pointLightOn[i];
                const char* walls[] = { "North", "South", "East", "West" };
                cout << "Wall Light[" << walls[i] << "] " << (pointLightOn[i] ? "ON" : "OFF") << "\n";
                keyProcessed[key] = true;
            }
        } else keyProcessed[key] = false;
    }

    // Ambient / Diffuse / Specular  ->  F5 F6 F7
    if (glfwGetKey(window, GLFW_KEY_F5) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_F5]) { ambientOn  = !ambientOn;  cout << "Ambient "  << (ambientOn  ? "ON" : "OFF") << "\n"; keyProcessed[GLFW_KEY_F5] = true; }
    } else keyProcessed[GLFW_KEY_F5] = false;
    if (glfwGetKey(window, GLFW_KEY_F6) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_F6]) { diffuseOn  = !diffuseOn;  cout << "Diffuse "  << (diffuseOn  ? "ON" : "OFF") << "\n"; keyProcessed[GLFW_KEY_F6] = true; }
    } else keyProcessed[GLFW_KEY_F6] = false;
    if (glfwGetKey(window, GLFW_KEY_F7) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_F7]) { specularOn = !specularOn; cout << "Specular " << (specularOn ? "ON" : "OFF") << "\n"; keyProcessed[GLFW_KEY_F7] = true; }
    } else keyProcessed[GLFW_KEY_F7] = false;

    // Split view  ->  F9
    if (glfwGetKey(window, GLFW_KEY_F9) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_F9]) { isSplitView = !isSplitView; cout << "Split View " << (isSplitView ? "ON" : "OFF") << "\n"; keyProcessed[GLFW_KEY_F9] = true; }
    } else keyProcessed[GLFW_KEY_F9] = false;
}

// ==========================================
// CALLBACKS
// ==========================================
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);
    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos; lastY = ypos;
    cameraYaw   += xoffset * 0.1f;
    cameraPitch += yoffset * 0.1f;
    if (cameraPitch >  89.0f) cameraPitch =  89.0f;
    if (cameraPitch < -89.0f) cameraPitch = -89.0f;
    glm::vec3 front;
    front.x = cos(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
    front.y = sin(glm::radians(cameraPitch));
    front.z = sin(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
    cameraFront = glm::normalize(front);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}