#include <cstddef>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "shader.h"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace std;
using namespace glm;

// ─── Function Prototypes ─────────────────────────────────────────────────────
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void processInput(GLFWwindow* window);
unsigned int loadTexture(char const* path);
void drawCube(unsigned int VAO, Shader& shader, glm::mat4 model, glm::vec3 color, unsigned int textureID = 0, glm::vec2 texScale = glm::vec2(1.0f, 1.0f));
void drawLightCube(unsigned int VAO, Shader& shader, glm::mat4 model, glm::vec3 color);
void drawGameHubScene(unsigned int VAO, Shader& shader, unsigned int poolTexture, unsigned int grassTexture);
void drawSkyAndSun(unsigned int VAO, Shader& lightCubeShader, glm::mat4 projection, glm::mat4 view, glm::vec3 eye);
void drawAllLights(unsigned int VAO, Shader& lightCubeShader, glm::mat4 projection, glm::mat4 view);
void setLightUniforms(Shader& shader);
// New geometry helpers (Surface of Revolution + GL_TRIANGLE_FAN)
void setupSphereGeometry();
void setupDiskGeometry();
void drawSphere(Shader& shader, glm::mat4 model, glm::vec3 color);
void drawDisk(Shader& shader, glm::mat4 model, glm::vec3 color);
// Pool table sub-functions
void drawPoolTable(unsigned int VAO, Shader& shader);
void drawPoolBalls(Shader& shader);
// Table tennis sub-functions
void drawTableTennis(unsigned int VAO, Shader& shader);
void drawTableTennisBalls(Shader& shader);
void drawTableTennisPaddles(unsigned int VAO, Shader& shader);
// Carrom board sub-functions
void drawCarromBoard(unsigned int VAO, Shader& shader);
void drawCarromPieces(Shader& shader);
void drawSofa(unsigned int VAO, Shader& shader, glm::mat4 model, glm::vec3 baseColor);

// ─── Settings ────────────────────────────────────────────────────────────────
const unsigned int SCR_WIDTH  = 1200;
const unsigned int SCR_HEIGHT = 800;

// ─── Camera ──────────────────────────────────────────────────────────────────
glm::vec3 cameraPos   = glm::vec3(0.0f, 4.0f, 8.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, -0.3f, -1.0f);
glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f, 0.0f);
bool  firstMouse  = true;
float cameraYaw   = -90.0f;
float cameraPitch = -20.0f;
float lastX = SCR_WIDTH  / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// ─── Light Sources ───────────────────────────────────────────────────────────
// 3 Spotlights above game stations; 4 point lights on walls
const glm::vec3 SPOT_POSITIONS[3] = {
    glm::vec3(-3.0f, 5.7f,  0.0f),   // Pool Table
    glm::vec3( 4.0f, 5.7f, -2.0f),   // Table Tennis
    glm::vec3( 4.0f, 5.7f,  3.0f)    // Carrom Board
};
const glm::vec3 POINT_POSITIONS[4] = {
    glm::vec3( 0.0f,  5.0f, -5.85f), // North wall
    glm::vec3( 0.0f,  5.0f,  5.85f), // South wall
    glm::vec3( 7.85f, 5.0f,  0.0f),  // East wall
    glm::vec3(-7.85f, 5.0f,  0.0f)   // West wall
};

// ─── Toggle States ───────────────────────────────────────────────────────────
bool spotLightOn[3]  = { true, true, true };
bool pointLightOn[4] = { true, true, true, true };
bool ambientOn   = true;
bool diffuseOn   = true;
bool specularOn  = true;
bool isSplitView = false;
bool keyProcessed[1024] = { false };
bool isDoorOpen = true;
float currentDoorAngle = 45.0f;
bool grassTextureEnabled = true;

// ─── Shading / Texture ───────────────────────────────────────────────────────
bool useGouraud      = false;
int  globalTextureMode = 1;   // 0=None, 1=Texture, 2=Blend
unsigned int grassTextureID;

// ─── Sphere / Disk Geometry (Surface of Revolution) ─────────────────────────
unsigned int sphereVAO, sphereVBO, sphereEBO;
int          sphereIndexCount = 0;
unsigned int diskVAO, diskVBO;
int          diskVertexCount  = 0;

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
    uniform vec3 viewPos;
    uniform vec2 texScale;

    void main() {
        FragPos     = vec3(model * vec4(aPos, 1.0));
        Normal      = mat3(transpose(inverse(model))) * aNormal;
        TexCoords   = aTexCoords * texScale;
        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
)";

// ==========================================
// 2. PHONG FRAGMENT SHADER (Per-Fragment)
// I = I_ambient + I_diffuse + I_specular
// I_a = k_a*L_a, I_d = k_d*L_d*max(L.N,0), I_s = k_s*L_s*max(R.V,0)^n
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

    struct DirLight {
        vec3 direction;
        vec3 ambient;
        vec3 diffuse;
        vec3 specular;
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
    uniform int  textureMode;

    uniform PointLight pointLights[NR_POINT_LIGHTS];
    uniform SpotLight  spotLights[NR_SPOT_LIGHTS];
    uniform DirLight   dirLight;
    uniform bool enablePoint[NR_POINT_LIGHTS];
    uniform bool enableSpot[NR_SPOT_LIGHTS];
    uniform bool enableDir;
    uniform bool enableAmbient;
    uniform bool enableDiffuse;
    uniform bool enableSpecular;
    uniform vec2 texScale;

    vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir) {
        vec3 lightDir = normalize(-light.direction);
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        vec3 ambient  = enableAmbient  ? light.ambient  : vec3(0.0);
        vec3 diffuse  = enableDiffuse  ? light.diffuse  * diff : vec3(0.0);
        vec3 specular = enableSpecular ? light.specular * spec : vec3(0.0);
        return (ambient + diffuse + specular);
    }

    vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
        vec3  lightDir   = normalize(light.position - fragPos);
        float diff       = max(dot(normal, lightDir), 0.0);
        vec3  reflectDir = reflect(-lightDir, normal);
        float spec       = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        float dist       = length(light.position - fragPos);
        float atten      = 1.0 / (light.constant + light.linear * dist + light.quadratic * dist * dist);
        vec3 ambient  = enableAmbient  ? light.ambient  * atten        : vec3(0.0);
        vec3 diffuse  = enableDiffuse  ? light.diffuse  * diff * atten : vec3(0.0);
        vec3 specular = enableSpecular ? light.specular * spec * atten : vec3(0.0);
        return ambient + diffuse + specular;
    }

    vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
        vec3  lightDir  = normalize(light.position - fragPos);
        float theta     = dot(lightDir, normalize(-light.direction));
        float epsilon   = light.cutOff - light.outerCutOff;
        float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
        float diff       = max(dot(normal, lightDir), 0.0);
        vec3  reflectDir = reflect(-lightDir, normal);
        float spec       = pow(max(dot(viewDir, reflectDir), 0.0), 64.0);
        float dist       = length(light.position - fragPos);
        float atten      = 1.0 / (light.constant + light.linear * dist + light.quadratic * dist * dist);
        vec3 ambient  = enableAmbient  ? light.ambient  * atten                   : vec3(0.0);
        vec3 diffuse  = enableDiffuse  ? light.diffuse  * diff * atten * intensity : vec3(0.0);
        vec3 specular = enableSpecular ? light.specular * spec * atten * intensity : vec3(0.0);
        return ambient + diffuse + specular;
    }

    void main() {
        vec3 norm    = normalize(Normal);
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 result  = vec3(0.0);
        if (enableDir) result += CalcDirLight(dirLight, norm, viewDir);
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

    struct DirLight {
        vec3 direction;
        vec3 ambient;
        vec3 diffuse;
        vec3 specular;
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
    uniform DirLight   dirLight;
    uniform bool enablePoint[NR_POINT_LIGHTS];
    uniform bool enableSpot[NR_SPOT_LIGHTS];
    uniform bool enableDir;
    uniform bool enableAmbient;
    uniform bool enableDiffuse;
    uniform bool enableSpecular;
    uniform vec2 texScale;

    vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir) {
        vec3 lightDir = normalize(-light.direction);
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        vec3 ambient  = enableAmbient  ? light.ambient  : vec3(0.0);
        vec3 diffuse  = enableDiffuse  ? light.diffuse  * diff : vec3(0.0);
        vec3 specular = enableSpecular ? light.specular * spec : vec3(0.0);
        return (ambient + diffuse + specular);
    }

    vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
        vec3  lightDir   = normalize(light.position - fragPos);
        float diff       = max(dot(normal, lightDir), 0.0);
        vec3  reflectDir = reflect(-lightDir, normal);
        float spec       = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        float dist       = length(light.position - fragPos);
        float atten      = 1.0 / (light.constant + light.linear * dist + light.quadratic * dist * dist);
        vec3 ambient  = enableAmbient  ? light.ambient  * atten        : vec3(0.0);
        vec3 diffuse  = enableDiffuse  ? light.diffuse  * diff * atten : vec3(0.0);
        vec3 specular = enableSpecular ? light.specular * spec * atten : vec3(0.0);
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
        vec3 diffuse  = enableDiffuse  ? light.diffuse  * diff * atten * intensity : vec3(0.0);
        vec3 specular = enableSpecular ? light.specular * spec * atten * intensity : vec3(0.0);
        return ambient + diffuse + specular;
    }

    void main() {
        vec3 FragPos = vec3(model * vec4(aPos, 1.0));
        vec3 Normal  = normalize(mat3(transpose(inverse(model))) * aNormal);
        TexCoords    = aTexCoords * texScale;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 result  = vec3(0.0);
        if (enableDir) result += CalcDirLight(dirLight, Normal, viewDir);
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
// 5. LIGHT CUBE SHADER (unlit emissive)
// ==========================================
const std::string vertexShaderLightCubeSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    out vec3 LocalPos;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    void main() {
        LocalPos = aPos;
        gl_Position = projection * view * model * vec4(aPos, 1.0);
    }
)";
const std::string fragmentShaderLightCubeSource = R"(
    #version 330 core
    out vec4 FragColor;
    in vec3 LocalPos;
    uniform vec3 objectColor;
    uniform bool isSun;
    uniform bool isSky;
    void main() {
        if (isSky) {
            float h = normalize(LocalPos).y;
            // Realistic sky gradient: deep blue zenith to light blue/white horizon
            vec3 skyTop = vec3(0.1f, 0.35f, 0.8f);
            vec3 skyBot = vec3(0.75f, 0.85f, 1.0f);
            vec3 res = mix(skyBot, skyTop, clamp(h * 0.8 + 0.2, 0.0, 1.0));
            FragColor = vec4(res, 1.0);
        } else if (isSun) {
            // Sun with a simple core and glow
            float dist = length(LocalPos);
            float glow = 1.0 - smoothstep(0.7, 1.0, dist);
            FragColor = vec4(objectColor, 1.0); 
        } else {
            FragColor = vec4(objectColor, 1.0);
        }
    }
)";

// ==========================================
// MAIN
// ==========================================
int main()
{
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

    Shader phongShader(vertexShaderPhongSource,     fragmentShaderPhongSource);
    Shader gouraudShader(vertexShaderGouraudSource, fragmentShaderGouraudSource);
    Shader lightCubeShader(vertexShaderLightCubeSource, fragmentShaderLightCubeSource);

    unsigned int poolStickerTexture = loadTexture("resources/Designer.png");
    grassTextureID = loadTexture("resources/grass.jpg");

    // Cube vertex data: positions, normals, tex coords
    float vertices[] = {
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

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // Pool ball spheres (Surface of Revolution) + pocket disks (GL_TRIANGLE_FAN)
    setupSphereGeometry();
    setupDiskGeometry();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // ── Render Loop ───────────────────────────────────────────────────────────
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
            glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)displayW / displayH, 0.1f, 400.0f); // increased far plane for sky
            glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
            drawSkyAndSun(VAO, lightCubeShader, proj, view, cameraPos);
            activeShader.use();
            activeShader.setMat4("projection", proj);
            activeShader.setMat4("view", view);
            drawGameHubScene(VAO, activeShader, poolStickerTexture, grassTextureID);
            drawAllLights(VAO, lightCubeShader, proj, view);
        }
        else {
            float ar = (float)(displayW / 2) / (float)(displayH / 2);
            glm::mat4 proj = glm::perspective(glm::radians(45.0f), ar, 0.1f, 400.0f);
            glm::mat4 view;

            glViewport(0, displayH / 2, displayW / 2, displayH / 2);
            glm::vec3 eye1 = glm::vec3(0.0f, 15.0f, 0.1f);
            view = glm::lookAt(eye1, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            drawSkyAndSun(VAO, lightCubeShader, proj, view, eye1);
            activeShader.use(); activeShader.setMat4("projection", proj); activeShader.setMat4("view", view);
            drawGameHubScene(VAO, activeShader, poolStickerTexture, grassTextureID);
            drawAllLights(VAO, lightCubeShader, proj, view);

            glViewport(displayW / 2, displayH / 2, displayW / 2, displayH / 2);
            glm::vec3 eye2 = glm::vec3(14.0f, 3.0f, 0.0f);
            view = glm::lookAt(eye2, glm::vec3(0.0f, 3.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            drawSkyAndSun(VAO, lightCubeShader, proj, view, eye2);
            activeShader.use(); activeShader.setMat4("view", view);
            drawGameHubScene(VAO, activeShader, poolStickerTexture, grassTextureID);
            drawAllLights(VAO, lightCubeShader, proj, view);

            glViewport(0, 0, displayW / 2, displayH / 2);
            glm::vec3 eye3 = glm::vec3(0.0f, 2.0f, 14.0f);
            view = glm::lookAt(eye3, glm::vec3(0.0f, 2.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            drawSkyAndSun(VAO, lightCubeShader, proj, view, eye3);
            activeShader.use(); activeShader.setMat4("view", view);
            drawGameHubScene(VAO, activeShader, poolStickerTexture, grassTextureID);
            drawAllLights(VAO, lightCubeShader, proj, view);

            glViewport(displayW / 2, 0, displayW / 2, displayH / 2);
            view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
            drawSkyAndSun(VAO, lightCubeShader, proj, view, cameraPos);
            activeShader.use(); activeShader.setMat4("view", view);
            drawGameHubScene(VAO, activeShader, poolStickerTexture, grassTextureID);
            drawAllLights(VAO, lightCubeShader, proj, view);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &sphereVAO);
    glDeleteBuffers(1, &sphereVBO);
    glDeleteBuffers(1, &sphereEBO);
    glDeleteVertexArrays(1, &diskVAO);
    glDeleteBuffers(1, &diskVBO);
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

    glm::vec3 bulbDiffuse  = glm::vec3(1.0f, 0.92f, 0.75f);
    glm::vec3 bulbSpecular = glm::vec3(1.0f, 0.95f, 0.85f);
    for (int i = 0; i < 3; i++) {
        std::string n = "spotLights[" + std::to_string(i) + "].";
        shader.setBool("enableSpot[" + std::to_string(i) + "]", spotLightOn[i]);
        shader.setVec3(n + "position",   SPOT_POSITIONS[i]);
        shader.setVec3(n + "direction",  glm::vec3(0.0f, -1.0f, 0.0f));
        shader.setVec3(n + "ambient",    glm::vec3(0.0f));
        shader.setVec3(n + "diffuse",    bulbDiffuse);
        shader.setVec3(n + "specular",   bulbSpecular);
        shader.setFloat(n + "constant",  1.0f);
        shader.setFloat(n + "linear",    0.07f);
        shader.setFloat(n + "quadratic", 0.017f);
        shader.setFloat(n + "cutOff",      glm::cos(glm::radians(18.0f)));
        shader.setFloat(n + "outerCutOff", glm::cos(glm::radians(25.0f)));
    }

    glm::vec3 wallDiffuse  = glm::vec3(0.75f, 0.80f, 0.85f);
    glm::vec3 wallSpecular = glm::vec3(0.50f, 0.55f, 0.60f);
    for (int i = 0; i < 4; i++) {
        std::string n = "pointLights[" + std::to_string(i) + "].";
        shader.setBool("enablePoint[" + std::to_string(i) + "]", pointLightOn[i]);
        shader.setVec3(n + "position",  POINT_POSITIONS[i]);
        shader.setVec3(n + "ambient",   glm::vec3(0.05f));
        shader.setVec3(n + "diffuse",   wallDiffuse);
        shader.setVec3(n + "specular",  wallSpecular);
        shader.setFloat(n + "constant",  1.0f);
        shader.setFloat(n + "linear",    0.045f);
        shader.setFloat(n + "quadratic", 0.0075f);
    }

    // ─── Sun (Directional Light) ──────────────────────────────────────────────
    shader.setBool("enableDir", true);
    shader.setVec3("dirLight.direction", glm::vec3(-0.2f, -1.0f, -0.3f));
    shader.setVec3("dirLight.ambient",   glm::vec3(0.25f, 0.25f, 0.22f)); // Warmer ambient
    shader.setVec3("dirLight.diffuse",   glm::vec3(1.0f, 0.98f, 0.85f));  // Brighter, warmer sun
    shader.setVec3("dirLight.specular",  glm::vec3(0.8f, 0.8f, 0.7f));    // Stronger specular
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
// DRAW HELPERS – Cube and Light Cube
// ==========================================
void drawCube(unsigned int VAO, Shader& shader, glm::mat4 model, glm::vec3 color, unsigned int textureID, glm::vec2 texScale) {
    shader.setMat4("model", model);
    shader.setVec3("objectColor", color);
    shader.setVec2("texScale", texScale);
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
    shader.setVec2("texScale", glm::vec2(1.0f, 1.0f));
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
}

// ==========================================
// SPHERE SETUP – Surface of Revolution (Requirements §2)
// x = r·cos(θ),  z = −r·sin(θ),  y = r·sin(φ)
// Unit sphere (radius=1); scale via model matrix when drawing.
// ==========================================
void setupSphereGeometry() {
    const float PI      = glm::pi<float>();
    const int   SECTORS = 20;  // longitude divisions
    const int   STACKS  = 14;  // latitude  divisions

    std::vector<float>        verts;
    std::vector<unsigned int> inds;
    verts.reserve((SECTORS + 1) * (STACKS + 1) * 8);
    inds.reserve(SECTORS * STACKS * 6);

    for (int i = 0; i <= STACKS; ++i) {
        float phi = PI / 2.0f - i * PI / (float)STACKS; // +π/2 → −π/2
        float y   = sinf(phi);
        float r   = cosf(phi);
        for (int j = 0; j <= SECTORS; ++j) {
            float theta = j * 2.0f * PI / (float)SECTORS;
            // Surface of Revolution formula (Requirements §2):
            float x = r * cosf(theta);
            float z = -r * sinf(theta);
            float s = (float)j / (float)SECTORS;
            float t = (float)i / (float)STACKS;
            // pos(x,y,z)  normal=pos for unit sphere  uv(s,t)
            verts.insert(verts.end(), { x, y, z,  x, y, z,  s, t });
        }
    }
    for (int i = 0; i < STACKS; ++i) {
        for (int j = 0; j < SECTORS; ++j) {
            unsigned int p1 = i * (SECTORS + 1) + j;
            unsigned int p2 = p1 + (SECTORS + 1);
            inds.insert(inds.end(), { p1, p2, p1+1,  p1+1, p2, p2+1 });
        }
    }
    sphereIndexCount = (int)inds.size();

    glGenVertexArrays(1, &sphereVAO);
    glGenBuffers(1, &sphereVBO);
    glGenBuffers(1, &sphereEBO);
    glBindVertexArray(sphereVAO);
    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(verts.size() * sizeof(float)), verts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(inds.size() * sizeof(unsigned int)), inds.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);
}

// ==========================================
// DISK SETUP – GL_TRIANGLE_FAN (Requirements §1)
// Unit disk in the XZ plane (y=0, normal=+Y).
// Scale via model matrix: vec3(radius, 1, radius) for a horizontal disk.
// ==========================================
void setupDiskGeometry() {
    const float PI       = glm::pi<float>();
    const int   SEGMENTS = 32;

    std::vector<float> verts;
    verts.reserve((SEGMENTS + 2) * 8);

    // Fan centre vertex
    verts.insert(verts.end(), { 0.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f,  0.5f, 0.5f });
    for (int i = 0; i <= SEGMENTS; ++i) {
        float theta = i * 2.0f * PI / (float)SEGMENTS;
        float x = cosf(theta), z = sinf(theta);
        verts.insert(verts.end(), { x, 0.0f, z,  0.0f, 1.0f, 0.0f,
                                    0.5f + 0.5f * x, 0.5f + 0.5f * z });
    }
    diskVertexCount = SEGMENTS + 2;

    glGenVertexArrays(1, &diskVAO);
    glGenBuffers(1, &diskVBO);
    glBindVertexArray(diskVAO);
    glBindBuffer(GL_ARRAY_BUFFER, diskVBO);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(verts.size() * sizeof(float)), verts.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);
}

void drawSphere(Shader& shader, glm::mat4 model, glm::vec3 color) {
    shader.setMat4("model", model);
    shader.setVec3("objectColor", color);
    shader.setVec2("texScale", glm::vec2(1.0f, 1.0f));
    shader.setInt("textureMode", 0);
    glBindVertexArray(sphereVAO);
    glDrawElements(GL_TRIANGLES, sphereIndexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

// model matrix must scale XZ by radius; Y scale =1 keeps disk flat
void drawDisk(Shader& shader, glm::mat4 model, glm::vec3 color) {
    shader.setMat4("model", model);
    shader.setVec3("objectColor", color);
    shader.setVec2("texScale", glm::vec2(1.0f, 1.0f));
    shader.setInt("textureMode", 0);
    glBindVertexArray(diskVAO);
    glDrawArrays(GL_TRIANGLE_FAN, 0, diskVertexCount);
    glBindVertexArray(0);
}

// ==========================================
// DRAW SKY AND SUN
// ==========================================
void drawSkyAndSun(unsigned int VAO, Shader& lightCubeShader, glm::mat4 projection, glm::mat4 view, glm::vec3 eye) {
    lightCubeShader.use();
    lightCubeShader.setMat4("projection", projection);
    lightCubeShader.setMat4("view", view);

    // ─── SKY ──────────────────────────────────────────────────────────────
    // Draw the sky first with depth writing disabled.
    // This ensures it stays in the background and avoids the "box" feel.
    glDepthMask(GL_FALSE);
    lightCubeShader.setBool("isSky", true);
    lightCubeShader.setBool("isSun", false);
    // Huge sphere for the sky, centered at the eye position to keep it distant.
    // Using a radius of 250 to ensure it's outside all scene objects.
    drawSphere(lightCubeShader, glm::scale(glm::translate(glm::mat4(1.0f), eye), glm::vec3(250.0f)), glm::vec3(1.0f));
    glDepthMask(GL_TRUE);

    // ─── SUN ──────────────────────────────────────────────────────────────
    lightCubeShader.setBool("isSky", false);
    lightCubeShader.setBool("isSun", true);
    // Placed far away in the sky. Positioned to match the light direction.
    glm::mat4 sunM = glm::translate(glm::mat4(1.0f), glm::vec3(50.0f, 150.0f, 70.0f));
    drawSphere(lightCubeShader, glm::scale(sunM, glm::vec3(12.0f)), glm::vec3(1.0f, 0.98f, 0.85f));
    lightCubeShader.setBool("isSun", false);
}

// ==========================================
// DRAW ALL LIGHT FIXTURES
// ==========================================
void drawAllLights(unsigned int VAO, Shader& lightCubeShader, glm::mat4 projection, glm::mat4 view) {
    lightCubeShader.use();
    lightCubeShader.setMat4("projection", projection);
    lightCubeShader.setMat4("view", view);
    lightCubeShader.setBool("isSky", false);
    lightCubeShader.setBool("isSun", false);

    glm::vec3 bulbColor    = glm::vec3(1.0f, 0.95f, 0.75f);
    glm::vec3 cordColor    = glm::vec3(0.15f, 0.12f, 0.10f);
    glm::vec3 bracketColor = glm::vec3(0.8f,  0.8f,  0.8f);

    for (int i = 0; i < 3; i++) {
        if (!spotLightOn[i]) continue;
        glm::vec3 pos = SPOT_POSITIONS[i];
        float cordLen = 6.0f - (pos.y + 0.13f);
        float cordMid = 6.0f - cordLen / 2.0f;
        drawLightCube(VAO, lightCubeShader,
            glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(pos.x, cordMid, pos.z)),
                       glm::vec3(0.025f, cordLen, 0.025f)), cordColor);
        drawLightCube(VAO, lightCubeShader,
            glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(pos.x, pos.y+0.13f, pos.z)),
                       glm::vec3(0.12f, 0.10f, 0.12f)), glm::vec3(0.2f));
        drawLightCube(VAO, lightCubeShader,
            glm::scale(glm::translate(glm::mat4(1.0f), pos), glm::vec3(0.18f)), bulbColor);
    }

    for (int i = 0; i < 4; i++) {
        if (!pointLightOn[i]) continue;
        glm::vec3 pos = POINT_POSITIONS[i];
        glm::mat4 backM, bulbM;
        if      (i == 0) { backM = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(pos.x,pos.y,-5.97f)), glm::vec3(0.35f,0.35f,0.05f)); bulbM = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(pos.x,pos.y,-5.78f)), glm::vec3(0.15f,0.15f,0.20f)); }
        else if (i == 1) { backM = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(pos.x,pos.y, 5.97f)), glm::vec3(0.35f,0.35f,0.05f)); bulbM = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(pos.x,pos.y, 5.78f)), glm::vec3(0.15f,0.15f,0.20f)); }
        else if (i == 2) { backM = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3( 7.97f,pos.y,pos.z)), glm::vec3(0.05f,0.35f,0.35f)); bulbM = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3( 7.78f,pos.y,pos.z)), glm::vec3(0.20f,0.15f,0.15f)); }
        else              { backM = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-7.97f,pos.y,pos.z)), glm::vec3(0.05f,0.35f,0.35f)); bulbM = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-7.78f,pos.y,pos.z)), glm::vec3(0.20f,0.15f,0.15f)); }
        drawLightCube(VAO, lightCubeShader, backM, bracketColor);
        drawLightCube(VAO, lightCubeShader, bulbM, bulbColor);
    }
}

// ==========================================
// POOL TABLE (Billiards) – Realistic Version
// Centre: world (−3, 0.8, 0)
// Playing surface: 3.0 (x-width) × 4.5 (z-length)
// 6 pockets: 4 corner + 2 side (dark disks on baize)
// ==========================================
void drawPoolTable(unsigned int VAO, Shader& shader) {
    const glm::vec3 C = glm::vec3(-3.0f, 0.8f, 0.0f);  // table centre

    const glm::vec3 mahogany  = glm::vec3(0.24f, 0.10f, 0.03f);
    const glm::vec3 feltPlay  = glm::vec3(0.05f, 0.42f, 0.15f);
    const glm::vec3 feltCush  = glm::vec3(0.07f, 0.35f, 0.10f);
    const glm::vec3 pocketClr = glm::vec3(0.03f, 0.03f, 0.03f);
    const glm::vec3 leatherCl = glm::vec3(0.08f, 0.04f, 0.01f);
    const glm::vec3 spotClr   = glm::vec3(0.72f, 0.72f, 0.62f);
    const glm::vec3 lineClr   = glm::vec3(0.48f, 0.48f, 0.38f);
    const glm::vec3 cueWood   = glm::vec3(0.80f, 0.62f, 0.38f);
    const glm::vec3 cueTip    = glm::vec3(0.22f, 0.42f, 0.72f);

    // ── Outer slate frame / support ──────────────────────────────────────────
    drawCube(VAO, shader,
        glm::scale(glm::translate(glm::mat4(1.0f), C + glm::vec3(0,-0.08f,0)),
        glm::vec3(3.55f, 0.12f, 5.1f)), mahogany);

    // ── Four turned legs + foot pads ─────────────────────────────────────────
    const glm::vec3 LO[4] = {
        {-1.42f,-0.55f,-2.1f}, {1.42f,-0.55f,-2.1f},
        {-1.42f,-0.55f, 2.1f}, {1.42f,-0.55f, 2.1f}
    };
    for (const auto& lo : LO) {
        drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+lo),             glm::vec3(0.22f,1.0f,0.22f)), mahogany);
        drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+lo+glm::vec3(0,-0.52f,0)), glm::vec3(0.30f,0.06f,0.30f)), mahogany);
    }

    // ── Side apron panels (below rail, between legs) ──────────────────────────
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3(-1.58f,-0.13f, 0)), glm::vec3(0.10f,0.55f,4.35f)), mahogany);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3( 1.58f,-0.13f, 0)), glm::vec3(0.10f,0.55f,4.35f)), mahogany);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3(0,-0.13f,-2.38f)), glm::vec3(3.15f,0.55f,0.10f)), mahogany);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3(0,-0.13f, 2.38f)), glm::vec3(3.15f,0.55f,0.10f)), mahogany);

    // ── Playing baize (felt surface) ─────────────────────────────────────────
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C), glm::vec3(3.0f,0.08f,4.5f)), feltPlay);

    // ── Cushion rails ────────────────────────────────────────────────────────
    // Long rails (x=±1.53): split at side pocket (gap at |z|<0.14)
    //  upper half centre z=-1.03, len 1.79;  lower half centre z=+1.03, len 1.79
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3(-1.53f,0.10f,-1.03f)), glm::vec3(0.17f,0.22f,1.79f)), feltCush);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3(-1.53f,0.10f, 1.03f)), glm::vec3(0.17f,0.22f,1.79f)), feltCush);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3( 1.53f,0.10f,-1.03f)), glm::vec3(0.17f,0.22f,1.79f)), feltCush);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3( 1.53f,0.10f, 1.03f)), glm::vec3(0.17f,0.22f,1.79f)), feltCush);

    // Short rails (z=±2.21): single continuous piece, pocket gaps at corners
    //  width 2.64 centred at x=0 (leaves ≈0.18 at each corner for pocket)
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3(0,0.10f,-2.21f)), glm::vec3(2.64f,0.22f,0.17f)), feltCush);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3(0,0.10f, 2.21f)), glm::vec3(2.64f,0.22f,0.17f)), feltCush);

    // ── Outer wooden top-cap rail ─────────────────────────────────────────────
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3(-1.67f,0.153f, 0)),    glm::vec3(0.25f,0.07f,4.78f)), mahogany);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3( 1.67f,0.153f, 0)),    glm::vec3(0.25f,0.07f,4.78f)), mahogany);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3(0, 0.153f,-2.38f)),    glm::vec3(3.26f,0.07f,0.26f)), mahogany);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3(0, 0.153f, 2.38f)),    glm::vec3(3.26f,0.07f,0.26f)), mahogany);

    // ── Pocket leather lining boxes ───────────────────────────────────────────
    // 4 corner pockets + 2 side pockets
    struct PXZ { float x, z; };
    const PXZ corners[4] = { {-1.47f,-2.13f},{1.47f,-2.13f},{-1.47f,2.13f},{1.47f,2.13f} };
    const PXZ sides[2]   = { {-1.55f, 0.0f}, {1.55f, 0.0f} };

    for (const auto& p : corners)
        drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f),
            C + glm::vec3(p.x, 0.01f, p.z)), glm::vec3(0.28f,0.10f,0.28f)), leatherCl);
    for (const auto& p : sides)
        drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f),
            C + glm::vec3(p.x, 0.01f, p.z)), glm::vec3(0.16f,0.10f,0.30f)), leatherCl);

    // ── Pocket holes – dark disks rendered via GL_TRIANGLE_FAN ───────────────
    // Disk is in XZ plane, normal=+Y. Scale (r, 1, r) = horizontal disk of radius r.
    const float feltTop = C.y + 0.042f;   // just above baize surface
    const float pocketR = 0.127f;

    for (const auto& p : corners)
        drawDisk(shader,
            glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x+p.x, feltTop, C.z+p.z)),
                       glm::vec3(pocketR, 1.0f, pocketR)), pocketClr);
    for (const auto& p : sides)
        drawDisk(shader,
            glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x+p.x, feltTop, C.z+p.z)),
                       glm::vec3(pocketR, 1.0f, pocketR)), pocketClr);

    // ── Sight spots on baize (head, centre, foot) ─────────────────────────────
    const float spotY = feltTop + 0.001f;
    auto drawSpot = [&](float wz) {
        drawDisk(shader,
            glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x, spotY, wz)),
                       glm::vec3(0.024f, 1.0f, 0.024f)), spotClr);
    };
    drawSpot(-1.125f);  // head spot  (1/4 from head end)
    drawSpot( 0.0f);    // centre spot
    drawSpot( 1.125f);  // foot spot  (1/4 from foot end)

    // Headstring: thin line across felt at head spot
    drawCube(VAO, shader,
        glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x, spotY+0.001f, -1.125f)),
        glm::vec3(2.9f, 0.004f, 0.005f)), lineClr);

    // ── Cue stick resting on long rail ────────────────────────────────────────
    glm::mat4 cueM = glm::translate(glm::mat4(1.0f), C + glm::vec3(0.92f, 0.20f, 0.5f));
    cueM = glm::rotate(cueM, glm::radians(12.0f),  glm::vec3(1.0f, 0.0f, 0.0f));
    cueM = glm::rotate(cueM, glm::radians(-10.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    drawCube(VAO, shader, glm::scale(cueM, glm::vec3(0.035f, 0.035f, 2.9f)), cueWood);
    // Chalk-blue tip
    glm::mat4 tipM = glm::translate(glm::mat4(1.0f), C + glm::vec3(0.75f, 0.24f, -0.85f));
    tipM = glm::rotate(tipM, glm::radians(-10.0f), glm::vec3(0.0f,1.0f,0.0f));
    drawCube(VAO, shader, glm::scale(tipM, glm::vec3(0.042f,0.042f,0.08f)), cueTip);
}

// ==========================================
// POOL BALLS – 16 Spheres (Surface of Revolution)
// Cue ball at head spot (z=−1.125)
// 15 object balls in standard 8-ball triangle rack at foot spot (z=+1.125)
// ==========================================
void drawPoolBalls(Shader& shader) {
    const float ballR = 0.07f;
    const float ballY = 0.8f + 0.04f + ballR;  // felt top + radius ≈ 0.91
    const float CX    = -3.0f;                  // table x-centre

    // Ball colours [0]=cue ball  [1..15]=object balls
    // Solids 1-7: rich pigment; 8=black; stripes 9-15: lighter/brighter hue
    const glm::vec3 BC[16] = {
        {0.96f,0.96f,0.94f},  // 0  cue ball  – near white
        {0.95f,0.85f,0.05f},  // 1  Yellow  solid
        {0.10f,0.25f,0.80f},  // 2  Blue    solid
        {0.88f,0.10f,0.07f},  // 3  Red     solid
        {0.50f,0.07f,0.55f},  // 4  Purple  solid
        {1.00f,0.42f,0.00f},  // 5  Orange  solid
        {0.10f,0.50f,0.15f},  // 6  Green   solid
        {0.52f,0.05f,0.05f},  // 7  Maroon  solid
        {0.07f,0.07f,0.07f},  // 8  Black
        {1.00f,0.95f,0.20f},  // 9  Yellow  stripe
        {0.30f,0.50f,0.95f},  // 10 Blue    stripe
        {1.00f,0.32f,0.25f},  // 11 Red     stripe
        {0.65f,0.20f,0.70f},  // 12 Purple  stripe
        {1.00f,0.60f,0.12f},  // 13 Orange  stripe
        {0.25f,0.70f,0.30f},  // 14 Green   stripe
        {0.72f,0.15f,0.15f},  // 15 Maroon  stripe
    };

    // ── Cue ball at head spot ─────────────────────────────────────────────────
    drawSphere(shader,
        glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(CX, ballY, -1.125f)), glm::vec3(ballR)),
        BC[0]);

    // ── Triangle rack at foot spot ────────────────────────────────────────────
    // Close-packing: dx = 2R+gap, dz = dx*sin(60°)
    const float dx  = 2.0f * ballR + 0.005f;    // ≈ 0.145
    const float dz  = dx * 0.8660254f;            // ≈ 0.1256
    const float RZ0 = 1.125f;                     // foot spot z

    // Standard 8-ball rack order:
    //  Row 0: [1]          (tip = 1-ball)
    //  Row 1: [2, 9]
    //  Row 2: [3, 8, 10]   (8 in centre)
    //  Row 3: [4, 14, 7, 12]
    //  Row 4: [11,13,15, 6, 5]  (stripe left corner, solid right corner)
    const int order[5][5] = {
        {  1, -1, -1, -1, -1 },
        {  2,  9, -1, -1, -1 },
        {  3,  8, 10, -1, -1 },
        {  4, 14,  7, 12, -1 },
        { 11, 13, 15,  6,  5 }
    };

    for (int row = 0; row < 5; ++row) {
        int   count = row + 1;
        float x0    = -(float)row * dx * 0.5f;  // leftmost ball x-offset from CX
        for (int col = 0; col < count; ++col) {
            int num = order[row][col];
            if (num < 0) continue;
            float px = CX + x0 + col * dx;
            float pz = RZ0 + row * dz;
            drawSphere(shader,
                glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(px, ballY, pz)), glm::vec3(ballR)),
                BC[num]);
        }
    }
}

// ==========================================
// SOFA – Realistic Furniture
// ==========================================
void drawSofa(unsigned int VAO, Shader& shader, glm::mat4 model, glm::vec3 baseColor) {
    glm::vec3 cushionColor = baseColor * 1.2f; // slightly lighter for cushions
    glm::vec3 darkColor = baseColor * 0.5f;    // darker for legs/base trim

    // model is assumed to define the center of the bottom base of the sofa.
    // Dimensions approximate: 2.4 wide, 1.0 deep, 1.0 high
    // Base platform
    drawCube(VAO, shader, model * glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.15f, 0.0f)), glm::vec3(2.4f, 0.3f, 1.0f)), baseColor);
    // Legs
    drawCube(VAO, shader, model * glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-1.1f, 0.05f, -0.4f)), glm::vec3(0.1f, 0.1f, 0.1f)), darkColor);
    drawCube(VAO, shader, model * glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3( 1.1f, 0.05f, -0.4f)), glm::vec3(0.1f, 0.1f, 0.1f)), darkColor);
    drawCube(VAO, shader, model * glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-1.1f, 0.05f,  0.4f)), glm::vec3(0.1f, 0.1f, 0.1f)), darkColor);
    drawCube(VAO, shader, model * glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3( 1.1f, 0.05f,  0.4f)), glm::vec3(0.1f, 0.1f, 0.1f)), darkColor);
    // Seat Cushions (left, right, middle - 3 cushions)
    drawCube(VAO, shader, model * glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-0.75f, 0.35f, 0.05f)), glm::vec3(0.7f, 0.15f, 0.8f)), cushionColor);
    drawCube(VAO, shader, model * glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3( 0.0f, 0.35f, 0.05f)), glm::vec3(0.7f, 0.15f, 0.8f)), cushionColor);
    drawCube(VAO, shader, model * glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3( 0.75f, 0.35f, 0.05f)), glm::vec3(0.7f, 0.15f, 0.8f)), cushionColor);
    // Backrest (covers full width between armrests)
    drawCube(VAO, shader, model * glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.65f, -0.4f)), glm::vec3(2.2f, 0.6f, 0.2f)), cushionColor);
    // Armrests (left and right)
    drawCube(VAO, shader, model * glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-1.15f, 0.5f, 0.0f)), glm::vec3(0.2f, 0.4f, 1.0f)), baseColor);
    drawCube(VAO, shader, model * glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3( 1.15f, 0.5f, 0.0f)), glm::vec3(0.2f, 0.4f, 1.0f)), baseColor);
}

// ==========================================
// SCENE DRAWING
// ==========================================
void drawGameHubScene(unsigned int VAO, Shader& shader, unsigned int poolTexture, unsigned int grassTexture) {
    glm::vec3 wallColor  = glm::vec3(0.85f, 0.88f, 0.90f);
    glm::vec3 floorColor = glm::vec3(0.38f, 0.38f, 0.38f);
    glm::vec3 roofColor  = glm::vec3(0.92f, 0.92f, 0.92f);
    glm::vec3 glassColor = glm::vec3(0.40f, 0.70f, 0.90f);
    glm::vec3 grassColor = glm::vec3(0.13f, 0.55f, 0.13f); // Forest Green
    glm::vec3 pathColor  = glm::vec3(0.50f, 0.45f, 0.40f); // Sandy brown path

    // 0. COMPOUND & PATHWAY ───────────────────────────────────────────────────
    // Large compound area (grass)
    unsigned int grassTexToUse = grassTextureEnabled ? grassTexture : 0;
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.06f, 0.0f)), glm::vec3(50.0f, 0.02f, 50.0f)), grassColor, grassTexToUse, glm::vec2(25.0f, 25.0f));

    // Pathway from the door (door is at z=6, x=0)
    // Starting at z=6.0 to z=25.0, centered at x=0
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.04f, 15.5f)), glm::vec3(2.0f, 0.03f, 19.0f)), pathColor);

    // 1. ROOM ─────────────────────────────────────────────────────────────────
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0,0,0)),       glm::vec3(16.0f,0.1f,12.0f)), floorColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0,6.0f,0)),    glm::vec3(16.0f,0.1f,12.0f)), roofColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0,3.0f,-6.0f)),glm::vec3(16.0f,6.0f,0.1f)), wallColor);

    // West wall with window
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-8.0f,1.0f, 0)),   glm::vec3(0.1f,2.0f,12.0f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-8.0f,5.0f, 0)),   glm::vec3(0.1f,2.0f,12.0f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-8.0f,3.0f,-4.0f)),glm::vec3(0.1f,2.0f,4.0f)),  wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-8.0f,3.0f, 4.0f)),glm::vec3(0.1f,2.0f,4.0f)),  wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-8.0f,3.0f, 0)),   glm::vec3(0.05f,2.0f,4.0f)), glassColor);

    // East wall with window
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(8.0f,1.0f, 0)),    glm::vec3(0.1f,2.0f,12.0f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(8.0f,5.0f, 0)),    glm::vec3(0.1f,2.0f,12.0f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(8.0f,3.0f,-4.0f)), glm::vec3(0.1f,2.0f,4.0f)),  wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(8.0f,3.0f, 4.0f)), glm::vec3(0.1f,2.0f,4.0f)),  wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(8.0f,3.0f, 0)),    glm::vec3(0.05f,2.0f,4.0f)), glassColor);

    // South wall with door
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-4.5f,3.0f,6.0f)), glm::vec3(7.0f,6.0f,0.1f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3( 4.5f,3.0f,6.0f)), glm::vec3(7.0f,6.0f,0.1f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3( 0.0f,4.5f,6.0f)), glm::vec3(2.0f,3.0f,0.1f)), wallColor);
    float targetAngle = isDoorOpen ? 75.0f : 0.0f;
    if (currentDoorAngle < targetAngle) {
        currentDoorAngle += 150.0f * deltaTime;
        if (currentDoorAngle > targetAngle) currentDoorAngle = targetAngle;
    } else if (currentDoorAngle > targetAngle) {
        currentDoorAngle -= 150.0f * deltaTime;
        if (currentDoorAngle < targetAngle) currentDoorAngle = targetAngle;
    }

    glm::mat4 door = glm::translate(glm::mat4(1.0f), glm::vec3(-1.0f,1.5f,6.0f));
    door = glm::rotate(door, glm::radians(currentDoorAngle), glm::vec3(0,1,0));
    glm::mat4 doorCenter = glm::translate(door, glm::vec3(1.0f,0,0));
    drawCube(VAO, shader, glm::scale(doorCenter, glm::vec3(2.0f,3.0f,0.05f)), glm::vec3(0.35f,0.18f,0.08f));

    // Door knob based on Surface of Revolution (Sphere)
    glm::mat4 knobL = glm::translate(doorCenter, glm::vec3(0.8f, -0.1f, -0.06f));
    drawSphere(shader, glm::scale(knobL, glm::vec3(0.04f, 0.04f, 0.04f)), glm::vec3(0.8f, 0.7f, 0.2f)); // Brass color inside
    glm::mat4 knobR = glm::translate(doorCenter, glm::vec3(0.8f, -0.1f, 0.06f));
    drawSphere(shader, glm::scale(knobR, glm::vec3(0.04f, 0.04f, 0.04f)), glm::vec3(0.8f, 0.7f, 0.2f)); // Brass color outside
    drawCube(VAO, shader, glm::scale(glm::translate(doorCenter, glm::vec3(0.8f, -0.1f, 0.0f)), glm::vec3(0.02f, 0.02f, 0.12f)), glm::vec3(0.7f, 0.6f, 0.1f));

    // 2. POOL (BILLIARDS) TABLE – realistic with pockets and balls ─────────────
    drawPoolTable(VAO, shader);
    drawPoolBalls(shader);

    // 3. TABLE TENNIS – realistic with net, painted lines, balls, paddles ─────
    drawTableTennis(VAO, shader);
    drawTableTennisBalls(shader);
    drawTableTennisPaddles(VAO, shader);

    // 4. CARROM BOARD ─────────────────────────────────────────────────────────
    drawCarromBoard(VAO, shader);
    drawCarromPieces(shader);

    // 5. SOFAS ────────────────────────────────────────────────────────────────
    // Sofa near Pool Table (West Wall)
    glm::mat4 poolSofaM = glm::translate(glm::mat4(1.0f), glm::vec3(-7.5f, 0.0f, 0.0f));
    poolSofaM = glm::rotate(poolSofaM, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    drawSofa(VAO, shader, poolSofaM, glm::vec3(0.22f, 0.20f, 0.20f)); // Dark grey leather

    // Sofa near Carrom Board (East Wall)
    glm::mat4 carromSofaM = glm::translate(glm::mat4(1.0f), glm::vec3(7.5f, 0.0f, 3.0f));
    carromSofaM = glm::rotate(carromSofaM, glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    drawSofa(VAO, shader, carromSofaM, glm::vec3(0.55f, 0.15f, 0.15f)); // Red leather
}

// ==========================================
// TABLE TENNIS – Realistic Ping-Pong Table
// Centre: world (4, 0.8, -2)
// Surface: 3.0 (x-length) × 1.6 (z-width)
// ==========================================
void drawTableTennis(unsigned int VAO, Shader& shader) {
    const glm::vec3 C  = glm::vec3(4.0f, 0.8f, -2.0f);
    const float ttTop  = C.y + 0.04f;  // surface y = 0.84

    const glm::vec3 tableBlue  = glm::vec3(0.08f, 0.25f, 0.65f);  // tournament blue baize
    const glm::vec3 metalFrame = glm::vec3(0.18f, 0.20f, 0.22f);  // dark aluminium frame
    const glm::vec3 lineWhite  = glm::vec3(0.95f, 0.95f, 0.92f);  // painted boundary lines
    const glm::vec3 netWhite   = glm::vec3(0.90f, 0.90f, 0.88f);  // net mesh
    const glm::vec3 netPost    = glm::vec3(0.28f, 0.28f, 0.30f);  // net post metal
    const glm::vec3 edgeWood   = glm::vec3(0.48f, 0.30f, 0.10f);  // wood edge banding

    // ── Playing surface ──────────────────────────────────────────────────────
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C), glm::vec3(3.0f,0.08f,1.6f)), tableBlue);

    // ── Wood edge banding (thin strip around table perimeter at surface level) 
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3( 0.0f,0.041f,-0.82f)), glm::vec3(3.02f,0.018f,0.04f)), edgeWood);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3( 0.0f,0.041f, 0.82f)), glm::vec3(3.02f,0.018f,0.04f)), edgeWood);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3(-1.52f,0.041f,0.0f)), glm::vec3(0.04f,0.018f,1.64f)), edgeWood);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3( 1.52f,0.041f,0.0f)), glm::vec3(0.04f,0.018f,1.64f)), edgeWood);

    // ── White boundary lines painted on baize (y = ttTop + 0.002f to avoid z-fighting) ──────────
    const float ly = ttTop + 0.002f;
    const float lineThick = 0.02f; // 2cm wide lines
    
    // Side lines (along length X, at width Z edges)
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x, ly, C.z - 0.79f)), glm::vec3(3.0f, 0.004f, lineThick)), lineWhite);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x, ly, C.z + 0.79f)), glm::vec3(3.0f, 0.004f, lineThick)), lineWhite);
    
    // End lines (along width Z, at length X edges)
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x - 1.49f, ly, C.z)), glm::vec3(lineThick, 0.004f, 1.6f)), lineWhite);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x + 1.49f, ly, C.z)), glm::vec3(lineThick, 0.004f, 1.6f)), lineWhite);
    
    // Centre divider line (DOUBLES line) - runs along length (X) in the middle of width (Z)
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x, ly, C.z)),       glm::vec3(3.0f, 0.004f, 0.008f)), lineWhite);
    
    // Middle line (under the net) - runs along width (Z) in the middle of length (X)
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x, ly, C.z)),       glm::vec3(0.012f, 0.004f, 1.6f)), lineWhite);

    // ── Underframe apron panels (metal, around perimeter below surface) ───────
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3( 0.0f,-0.065f,-0.79f)), glm::vec3(2.96f,0.10f,0.040f)), metalFrame);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3( 0.0f,-0.065f, 0.79f)), glm::vec3(2.96f,0.10f,0.040f)), metalFrame);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3(-1.49f,-0.065f,0.0f)), glm::vec3(0.040f,0.10f,1.58f)), metalFrame);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3( 1.49f,-0.065f,0.0f)), glm::vec3(0.040f,0.10f,1.58f)), metalFrame);

    // ── Four adjustable legs ──────────────────────────────────────────────────
    const glm::vec3 LP[4] = {
        {-1.38f,-0.45f,-0.72f}, { 1.38f,-0.45f,-0.72f},
        {-1.38f,-0.45f, 0.72f}, { 1.38f,-0.45f, 0.72f}
    };
    for (const auto& lp : LP)
        drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+lp), glm::vec3(0.055f,0.90f,0.055f)), metalFrame);

    // Leg cross-bar supports (for structural stability)
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3( 0.0f,-0.66f,-0.72f)), glm::vec3(2.76f,0.040f,0.040f)), metalFrame);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3( 0.0f,-0.66f, 0.72f)), glm::vec3(2.76f,0.040f,0.040f)), metalFrame);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3(-1.38f,-0.66f,0.0f)), glm::vec3(0.040f,0.040f,1.44f)), metalFrame);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C+glm::vec3( 1.38f,-0.66f,0.0f)), glm::vec3(0.040f,0.040f,1.44f)), metalFrame);

    // ── Net structure (15.25 cm high → 0.15 world units) ─────────────────────
    const float netH   = 0.15f;
    const float netMY  = ttTop + netH * 0.5f;   // net vertical midpoint
    const float netTY  = ttTop + netH;           // net top

    // Net posts (4 cm outside table edge on each side)
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x, netMY, C.z-0.84f)), glm::vec3(0.030f,netH+0.02f,0.030f)), netPost);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x, netMY, C.z+0.84f)), glm::vec3(0.030f,netH+0.02f,0.030f)), netPost);
    // Post clamp at table z-edge
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x, ttTop+0.012f, C.z-0.81f)), glm::vec3(0.055f,0.042f,0.042f)), netPost);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x, ttTop+0.012f, C.z+0.81f)), glm::vec3(0.055f,0.042f,0.042f)), netPost);
    // Top cord (runs across z-width between posts)
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x,netTY,C.z)), glm::vec3(0.008f,0.010f,1.72f)), netPost);
    // Horizontal net cords (mesh appearance, each running along z)
    for (int k = 0; k < 5; ++k) {
        float hy = ttTop + 0.018f + k * 0.026f;
        drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x,hy,C.z)), glm::vec3(0.007f,0.007f,1.68f)), netWhite);
    }
    // Vertical net cords (positioned at intervals along z-width)
    for (int k = -6; k <= 6; ++k) {
        float vz = C.z + k * 0.133f;
        drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x,netMY+0.01f,vz)), glm::vec3(0.007f,netH-0.02f,0.007f)), netWhite);
    }
}

// ==========================================
// TABLE TENNIS BALLS – 2 white ping-pong balls (spheres)
// Radius 0.04 world-units (≈ real 40 mm diameter)
// ==========================================
void drawTableTennisBalls(Shader& shader) {
    const glm::vec3 C  = glm::vec3(4.0f, 0.8f, -2.0f);
    const float ttTop  = C.y + 0.04f;
    const float ballR  = 0.04f;
    const float ballY  = ttTop + ballR;   // centre of ball sitting on surface
    const glm::vec3 ballClr = glm::vec3(0.97f, 0.97f, 0.95f); // white

    // Ball 1 – player 1 side (positive-x half, near side rail)
    drawSphere(shader,
        glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x+0.80f, ballY, C.z+0.50f)), glm::vec3(ballR)),
        ballClr);
    // Ball 2 – player 2 side (negative-x half)
    drawSphere(shader,
        glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x-0.70f, ballY, C.z-0.45f)), glm::vec3(ballR)),
        ballClr);
}

// ==========================================
// TABLE TENNIS PADDLES – 2 paddles lying flat on the table
// Each paddle: circular disk face (GL_TRIANGLE_FAN) + wood handle (cube)
// ==========================================
void drawTableTennisPaddles(unsigned int VAO, Shader& shader) {
    const glm::vec3 C  = glm::vec3(4.0f, 0.8f, -2.0f);
    const float ttTop  = C.y + 0.04f;
    const float padY   = ttTop + 0.006f;   // paddle face sits on table surface

    const glm::vec3 redRubber  = glm::vec3(0.82f, 0.07f, 0.07f);  // red rubber face
    const glm::vec3 blackRubber= glm::vec3(0.10f, 0.08f, 0.08f);  // black rubber back
    const glm::vec3 handleWood = glm::vec3(0.52f, 0.30f, 0.10f);  // wood handle
    const glm::vec3 handleEdge = glm::vec3(0.35f, 0.20f, 0.06f);  // handle edge strip

    // ── PADDLE 1 (player-1 side: positive-x, positive-z quadrant) ────────────
    {
        const float px = C.x + 0.82f, pz = C.z + 0.38f;
        // Red disk face (top surface)
        drawDisk(shader,
            glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(px, padY, pz)),
                       glm::vec3(0.160f, 1.0f, 0.160f)), redRubber);
        // Black disk back (slightly below)
        drawDisk(shader,
            glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(px, padY-0.003f, pz)),
                       glm::vec3(0.158f, 1.0f, 0.158f)), blackRubber);
        // Wood handle (extends in +z from edge of blade)
        drawCube(VAO, shader,
            glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(px, padY-0.006f, pz+0.16f+0.11f)),
                       glm::vec3(0.055f, 0.022f, 0.22f)), handleWood);
        // Handle edge trim
        drawCube(VAO, shader,
            glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(px, padY-0.006f, pz+0.16f+0.11f)),
                       glm::vec3(0.060f, 0.006f, 0.22f)), handleEdge);
    }

    // ── PADDLE 2 (player-2 side: negative-x, negative-z quadrant) ────────────
    {
        const float px = C.x - 0.82f, pz = C.z - 0.38f;
        // Red face
        drawDisk(shader,
            glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(px, padY, pz)),
                       glm::vec3(0.160f, 1.0f, 0.160f)), redRubber);
        // Black back
        drawDisk(shader,
            glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(px, padY-0.003f, pz)),
                       glm::vec3(0.158f, 1.0f, 0.158f)), blackRubber);
        // Handle extends in -z from blade edge
        drawCube(VAO, shader,
            glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(px, padY-0.006f, pz-0.16f-0.11f)),
                       glm::vec3(0.055f, 0.022f, 0.22f)), handleWood);
        drawCube(VAO, shader,
            glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(px, padY-0.006f, pz-0.16f-0.11f)),
                       glm::vec3(0.060f, 0.006f, 0.22f)), handleEdge);
    }
}

// ==========================================
// CARROM BOARD – Realistic version
// Surface: smooth plywood with lines and 4 pockets (disks)
// ==========================================
void drawCarromBoard(unsigned int VAO, Shader& shader) {
    const glm::vec3 C = glm::vec3(4.0f, 0.6f, 3.0f);

    const glm::vec3 plywood    = glm::vec3(0.92f, 0.82f, 0.68f);
    const glm::vec3 frameWood  = glm::vec3(0.35f, 0.18f, 0.08f); 
    const glm::vec3 pocketClr  = glm::vec3(0.04f, 0.04f, 0.04f);
    const glm::vec3 lineClr    = glm::vec3(0.15f, 0.15f, 0.15f);
    const glm::vec3 redLine    = glm::vec3(0.70f, 0.15f, 0.15f);
    const glm::vec3 legWood    = glm::vec3(0.20f, 0.10f, 0.05f);

    // 4 Legs
    float legO = 0.62f;
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C + glm::vec3( legO, -0.3f,  legO)), glm::vec3(0.08f, 0.6f, 0.08f)), legWood);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C + glm::vec3(-legO, -0.3f,  legO)), glm::vec3(0.08f, 0.6f, 0.08f)), legWood);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C + glm::vec3( legO, -0.3f, -legO)), glm::vec3(0.08f, 0.6f, 0.08f)), legWood);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C + glm::vec3(-legO, -0.3f, -legO)), glm::vec3(0.08f, 0.6f, 0.08f)), legWood);

    // Surface
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C), glm::vec3(1.2f, 0.05f, 1.2f)), plywood);

    // Frame (sides)
    float frH = 0.12f;
    float frY = 0.035f;
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C + glm::vec3(0, frY, -0.65f)), glm::vec3(1.4f, frH, 0.1f)), frameWood);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C + glm::vec3(0, frY,  0.65f)), glm::vec3(1.4f, frH, 0.1f)), frameWood);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C + glm::vec3(-0.65f, frY, 0)), glm::vec3(0.1f, frH, 1.2f)), frameWood);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), C + glm::vec3( 0.65f, frY, 0)), glm::vec3(0.1f, frH, 1.2f)), frameWood);

    // Pocket holes
    float pOffset = 0.535f;
    float pocketR = 0.058f;
    float fY = C.y + 0.026f; // slightly above surface
    drawDisk(shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x +  pOffset, fY, C.z +  pOffset)), glm::vec3(pocketR, 1.0f, pocketR)), pocketClr);
    drawDisk(shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x + -pOffset, fY, C.z +  pOffset)), glm::vec3(pocketR, 1.0f, pocketR)), pocketClr);
    drawDisk(shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x +  pOffset, fY, C.z + -pOffset)), glm::vec3(pocketR, 1.0f, pocketR)), pocketClr);
    drawDisk(shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x + -pOffset, fY, C.z + -pOffset)), glm::vec3(pocketR, 1.0f, pocketR)), pocketClr);

    // Center circles
    drawDisk(shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x, fY, C.z)), glm::vec3(0.18f, 1.0f, 0.18f)), lineClr);
    drawDisk(shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x, fY+0.0001f, C.z)), glm::vec3(0.17f, 1.0f, 0.17f)), plywood);
    // Inner small center point
    drawDisk(shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x, fY+0.0002f, C.z)), glm::vec3(0.015f, 1.0f, 0.015f)), redLine);

    // Baselines
    float lineDist = 0.4f;
    float lineDistIn = 0.35f;
    float lineLen = 0.8f;
    float fLine = C.y + 0.0255f;
    
    // Outer baselines
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x, fLine, C.z + lineDist)), glm::vec3(lineLen, 0.001f, 0.006f)), lineClr);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x, fLine, C.z - lineDist)), glm::vec3(lineLen, 0.001f, 0.006f)), lineClr);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x + lineDist, fLine, C.z)), glm::vec3(0.006f, 0.001f, lineLen)), lineClr);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x - lineDist, fLine, C.z)), glm::vec3(0.006f, 0.001f, lineLen)), lineClr);
    
    // Inner baselines
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x, fLine, C.z + lineDistIn)), glm::vec3(lineLen, 0.001f, 0.002f)), lineClr);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x, fLine, C.z - lineDistIn)), glm::vec3(lineLen, 0.001f, 0.002f)), lineClr);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x + lineDistIn, fLine, C.z)), glm::vec3(0.002f, 0.001f, lineLen)), lineClr);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x - lineDistIn, fLine, C.z)), glm::vec3(0.002f, 0.001f, lineLen)), lineClr);

    // End circles (base circles)
    float baseCircOffset = lineLen / 2.0f;
    float midDist = (lineDist + lineDistIn) / 2.0f;
    float circR = 0.038f;
    float fCirc = C.y + 0.0258f;
    const glm::vec2 circCoords[8] = {
        {baseCircOffset, midDist}, {-baseCircOffset, midDist},
        {baseCircOffset, -midDist}, {-baseCircOffset, -midDist},
        {midDist, baseCircOffset}, {midDist, -baseCircOffset},
        {-midDist, baseCircOffset}, {-midDist, -baseCircOffset}
    };
    for (int i = 0; i < 8; i++) {
        drawDisk(shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x + circCoords[i].x, fCirc, C.z + circCoords[i].y)), glm::vec3(circR, 1.0f, circR)), redLine);
        drawDisk(shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x + circCoords[i].x, fCirc+0.0001f, C.z + circCoords[i].y)), glm::vec3(circR*0.7f, 1.0f, circR*0.7f)), plywood);
    }
    
    // Diagonal lines
    glm::mat4 m1 = glm::translate(glm::mat4(1.0f), glm::vec3(C.x + 0.36f, fLine, C.z + 0.36f));
    m1 = glm::rotate(m1, glm::radians(45.0f), glm::vec3(0, 1, 0));
    drawCube(VAO, shader, glm::scale(m1, glm::vec3(0.18f, 0.001f, 0.004f)), lineClr);

    glm::mat4 m2 = glm::translate(glm::mat4(1.0f), glm::vec3(C.x - 0.36f, fLine, C.z + 0.36f));
    m2 = glm::rotate(m2, glm::radians(-45.0f), glm::vec3(0, 1, 0));
    drawCube(VAO, shader, glm::scale(m2, glm::vec3(0.18f, 0.001f, 0.004f)), lineClr);

    glm::mat4 m3 = glm::translate(glm::mat4(1.0f), glm::vec3(C.x + 0.36f, fLine, C.z - 0.36f));
    m3 = glm::rotate(m3, glm::radians(-45.0f), glm::vec3(0, 1, 0));
    drawCube(VAO, shader, glm::scale(m3, glm::vec3(0.18f, 0.001f, 0.004f)), lineClr);

    glm::mat4 m4 = glm::translate(glm::mat4(1.0f), glm::vec3(C.x - 0.36f, fLine, C.z - 0.36f));
    m4 = glm::rotate(m4, glm::radians(45.0f), glm::vec3(0, 1, 0));
    drawCube(VAO, shader, glm::scale(m4, glm::vec3(0.18f, 0.001f, 0.004f)), lineClr);
}

// ==========================================
// CARROM PIECES
// 19 men (9 w, 9 b, 1 r) + 1 striker
// ==========================================
void drawCarromPieces(Shader& shader) {
    const glm::vec3 C = glm::vec3(4.0f, 0.627f, 3.0f); // just above surface
    
    const float rMan = 0.032f;
    const float hMan = 0.008f;
    const float rStr = 0.044f;
    const float hStr = 0.010f;

    const glm::vec3 wClr = glm::vec3(0.95f, 0.90f, 0.80f);
    const glm::vec3 bClr = glm::vec3(0.12f, 0.12f, 0.12f);
    const glm::vec3 rClr = glm::vec3(0.85f, 0.15f, 0.20f);
    const glm::vec3 sClr = glm::vec3(0.85f, 0.90f, 0.95f); // Ivory/White Striker

    auto drawPiece = [&](float x, float z, float rad, float hScale, glm::vec3 color) {
        drawSphere(shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(x, C.y + hScale, z)), glm::vec3(rad, hScale, rad)), color);
    };

    // 1. Queen (Center)
    drawPiece(C.x, C.z, rMan, hMan, rClr);

    // 2. Inner circle (6 pieces: 3 white, 3 black alternating)
    float d1 = rMan * 2.05f;
    for (int i = 0; i < 6; ++i) {
        float angle = glm::radians((float)i * 60.0f);
        drawPiece(C.x + d1 * cos(angle), C.z + d1 * sin(angle), rMan, hMan, (i % 2 == 0) ? wClr : bClr);
    }

    // 3. Outer circle (12 pieces: 6 white, 6 black alternating)
    float d2 = rMan * 3.9f;
    for (int i = 0; i < 12; ++i) {
        float angle = glm::radians((float)i * 30.0f + 15.0f);
        drawPiece(C.x + d2 * cos(angle), C.z + d2 * sin(angle), rMan, hMan, (i % 2 == 0) ? bClr : wClr); 
    }

    // 4. Striker 
    float strikerZ = C.z + 0.375f; 
    drawPiece(C.x, strikerZ, rStr, hStr, sClr);

    // Striker design (colorful rings)
    drawDisk(shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x, C.y + 2.0f*hStr + 0.001f, strikerZ)), glm::vec3(rStr*0.55f, 1.0f, rStr*0.55f)), glm::vec3(0.9f, 0.2f, 0.2f));
    drawDisk(shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(C.x, C.y + 2.0f*hStr + 0.0015f, strikerZ)), glm::vec3(rStr*0.35f, 1.0f, rStr*0.35f)), sClr);
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

    // Shading mode
    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_G]) { useGouraud = true;  cout << "Gouraud Shading\n"; keyProcessed[GLFW_KEY_G] = true; }
    } else keyProcessed[GLFW_KEY_G] = false;
    if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_H]) { useGouraud = false; cout << "Phong Shading\n";   keyProcessed[GLFW_KEY_H] = true; }
    } else keyProcessed[GLFW_KEY_H] = false;

    // Texture mode
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_Z]) { globalTextureMode = 0; cout << "Texture OFF\n";   keyProcessed[GLFW_KEY_Z] = true; }
    } else keyProcessed[GLFW_KEY_Z] = false;
    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_X]) { globalTextureMode = 1; cout << "Texture ON\n";    keyProcessed[GLFW_KEY_X] = true; }
    } else keyProcessed[GLFW_KEY_X] = false;
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_C]) { globalTextureMode = 2; cout << "Texture BLEND\n"; keyProcessed[GLFW_KEY_C] = true; }
    } else keyProcessed[GLFW_KEY_C] = false;

    // Spotlights: keys 1-3
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
    // Wall point lights: keys 4-7
    for (int i = 0; i < 4; i++) {
        int key = GLFW_KEY_4 + i;
        if (glfwGetKey(window, key) == GLFW_PRESS) {
            if (!keyProcessed[key]) {
                pointLightOn[i] = !pointLightOn[i];
                const char* walls[] = { "North","South","East","West" };
                cout << "Wall Light[" << walls[i] << "] " << (pointLightOn[i] ? "ON" : "OFF") << "\n";
                keyProcessed[key] = true;
            }
        } else keyProcessed[key] = false;
    }

    // Ambient/Diffuse/Specular: F5-F7
    if (glfwGetKey(window, GLFW_KEY_F5) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_F5]) { ambientOn  = !ambientOn;  cout << "Ambient "  << (ambientOn  ? "ON":"OFF") << "\n"; keyProcessed[GLFW_KEY_F5] = true; }
    } else keyProcessed[GLFW_KEY_F5] = false;
    if (glfwGetKey(window, GLFW_KEY_F6) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_F6]) { diffuseOn  = !diffuseOn;  cout << "Diffuse "  << (diffuseOn  ? "ON":"OFF") << "\n"; keyProcessed[GLFW_KEY_F6] = true; }
    } else keyProcessed[GLFW_KEY_F6] = false;
    if (glfwGetKey(window, GLFW_KEY_F7) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_F7]) { specularOn = !specularOn; cout << "Specular " << (specularOn ? "ON":"OFF") << "\n"; keyProcessed[GLFW_KEY_F7] = true; }
    } else keyProcessed[GLFW_KEY_F7] = false;

    // Split view: F9
    if (glfwGetKey(window, GLFW_KEY_F9) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_F9]) { isSplitView = !isSplitView; cout << "Split View " << (isSplitView ? "ON":"OFF") << "\n"; keyProcessed[GLFW_KEY_F9] = true; }
    } else keyProcessed[GLFW_KEY_F9] = false;

    // Door toggle: 0 (Zero)
    if (glfwGetKey(window, GLFW_KEY_0) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_0]) { isDoorOpen = !isDoorOpen; cout << "Door " << (isDoorOpen ? "OPEN":"CLOSED") << "\n"; keyProcessed[GLFW_KEY_0] = true; }
    } else keyProcessed[GLFW_KEY_0] = false;

    // Grass Texture toggle: 9
    if (glfwGetKey(window, GLFW_KEY_9) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_9]) { grassTextureEnabled = !grassTextureEnabled; cout << "Grass Texture " << (grassTextureEnabled ? "ON":"OFF") << "\n"; keyProcessed[GLFW_KEY_9] = true; }
    } else keyProcessed[GLFW_KEY_9] = false;
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