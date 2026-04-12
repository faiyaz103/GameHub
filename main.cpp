#include <cstddef> 
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "shader.h"
#include <iostream>

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
void drawGameHubScene(unsigned int VAO, Shader& shader, unsigned int poolTexture); // Removed wallTexture

// --- Settings ---
const unsigned int SCR_WIDTH = 1200;
const unsigned int SCR_HEIGHT = 800;

// --- Camera ---
glm::vec3 cameraPos = glm::vec3(0.0f, 4.0f, 6.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, -0.3f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
bool firstMouse = true;
float cameraYaw = -90.0f;
float cameraPitch = -20.0f;
float lastX = SCR_WIDTH / 2.0;
float lastY = SCR_HEIGHT / 2.0;
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// --- State Toggles ---
bool dirLightOn = true; bool pointLightOn = true; bool spotLightOn = true;
bool ambientOn = true; bool diffuseOn = true; bool specularOn = true;
bool isSplitView = false;
bool keyProcessed[1024] = { false };

// --- Shading & Texture Mode Toggles ---
bool useGouraud = false; // Default to Phong
int globalTextureMode = 1; // 0: None, 1: Texture, 2: Blend

// ==========================================
// 1. PHONG SHADERS (Per-Fragment Lighting)
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
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = mat3(transpose(inverse(model))) * aNormal; 
        TexCoords = aTexCoords;
        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
)";

const std::string fragmentShaderPhongSource = R"(
    #version 330 core
    out vec4 FragColor;

    struct DirLight { vec3 direction; vec3 ambient; vec3 diffuse; vec3 specular; };
    struct PointLight { vec3 position; float constant; float linear; float quadratic; vec3 ambient; vec3 diffuse; vec3 specular; };
    struct SpotLight { vec3 position; vec3 direction; float cutOff; float outerCutOff; float constant; float linear; float quadratic; vec3 ambient; vec3 diffuse; vec3 specular; };

    in vec3 FragPos;
    in vec3 Normal;
    in vec2 TexCoords;

    uniform vec3 viewPos;
    uniform vec3 objectColor;
    uniform sampler2D diffuseMap;
    uniform int textureMode; // 0=None, 1=Tex, 2=Blend

    uniform DirLight dirLight; uniform PointLight pointLight; uniform SpotLight spotLight;
    uniform bool enableDir; uniform bool enablePoint; uniform bool enableSpot;
    uniform bool enableAmbient; uniform bool enableDiffuse; uniform bool enableSpecular;

    vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir) {
        vec3 lightDir = normalize(-light.direction);
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        return (enableAmbient ? light.ambient : vec3(0.0)) + 
               (enableDiffuse ? light.diffuse * diff : vec3(0.0)) + 
               (enableSpecular ? light.specular * spec : vec3(0.0));
    }
    vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
        vec3 lightDir = normalize(light.position - fragPos);
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        float distance = length(light.position - fragPos);
        float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));    
        return (enableAmbient ? light.ambient * attenuation : vec3(0.0)) + 
               (enableDiffuse ? light.diffuse * diff * attenuation : vec3(0.0)) + 
               (enableSpecular ? light.specular * spec * attenuation : vec3(0.0));
    }
    vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
        vec3 lightDir = normalize(light.position - fragPos);
        float theta = dot(lightDir, normalize(-light.direction));
        if(theta > light.cutOff) {       
            float diff = max(dot(normal, lightDir), 0.0);
            vec3 reflectDir = reflect(-lightDir, normal);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
            float distance = length(light.position - fragPos);
            float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));    
            return (enableAmbient ? light.ambient * attenuation : vec3(0.0)) + 
                   (enableDiffuse ? light.diffuse * diff * attenuation : vec3(0.0)) + 
                   (enableSpecular ? light.specular * spec * attenuation : vec3(0.0));
        } else { return vec3(0.0); }
    }

    void main() {
        vec3 norm = normalize(Normal);
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 result = vec3(0.0);

        if(enableDir) result += CalcDirLight(dirLight, norm, viewDir);
        if(enablePoint) result += CalcPointLight(pointLight, norm, FragPos, viewDir);
        if(enableSpot) result += CalcSpotLight(spotLight, norm, FragPos, viewDir);

        vec4 texColor;
        if(textureMode == 0) texColor = vec4(objectColor, 1.0);
        else if(textureMode == 1) texColor = texture(diffuseMap, TexCoords);
        else if(textureMode == 2) texColor = mix(vec4(objectColor, 1.0), texture(diffuseMap, TexCoords), 0.5);

        FragColor = vec4(result * texColor.rgb, texColor.a);
    }
)";

// ==========================================
// 2. GOURAUD SHADERS (Per-Vertex Lighting)
// ==========================================
const std::string vertexShaderGouraudSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in vec2 aTexCoords;

    out vec3 LightingColor; 
    out vec2 TexCoords;

    struct DirLight { vec3 direction; vec3 ambient; vec3 diffuse; vec3 specular; };
    struct PointLight { vec3 position; float constant; float linear; float quadratic; vec3 ambient; vec3 diffuse; vec3 specular; };
    struct SpotLight { vec3 position; vec3 direction; float cutOff; float outerCutOff; float constant; float linear; float quadratic; vec3 ambient; vec3 diffuse; vec3 specular; };

    uniform mat4 model; uniform mat4 view; uniform mat4 projection;
    uniform vec3 viewPos;
    
    uniform DirLight dirLight; uniform PointLight pointLight; uniform SpotLight spotLight;
    uniform bool enableDir; uniform bool enablePoint; uniform bool enableSpot;
    uniform bool enableAmbient; uniform bool enableDiffuse; uniform bool enableSpecular;

    vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir) {
        vec3 lightDir = normalize(-light.direction);
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        return (enableAmbient ? light.ambient : vec3(0.0)) + 
               (enableDiffuse ? light.diffuse * diff : vec3(0.0)) + 
               (enableSpecular ? light.specular * spec : vec3(0.0));
    }
    vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
        vec3 lightDir = normalize(light.position - fragPos);
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        float distance = length(light.position - fragPos);
        float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));    
        return (enableAmbient ? light.ambient * attenuation : vec3(0.0)) + 
               (enableDiffuse ? light.diffuse * diff * attenuation : vec3(0.0)) + 
               (enableSpecular ? light.specular * spec * attenuation : vec3(0.0));
    }
    vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
        vec3 lightDir = normalize(light.position - fragPos);
        float theta = dot(lightDir, normalize(-light.direction));
        if(theta > light.cutOff) {       
            float diff = max(dot(normal, lightDir), 0.0);
            vec3 reflectDir = reflect(-lightDir, normal);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
            float distance = length(light.position - fragPos);
            float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));    
            return (enableAmbient ? light.ambient * attenuation : vec3(0.0)) + 
                   (enableDiffuse ? light.diffuse * diff * attenuation : vec3(0.0)) + 
                   (enableSpecular ? light.specular * spec * attenuation : vec3(0.0));
        } else { return vec3(0.0); }
    }

    void main() {
        vec3 FragPos = vec3(model * vec4(aPos, 1.0));
        vec3 Normal = normalize(mat3(transpose(inverse(model))) * aNormal); 
        TexCoords = aTexCoords;

        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 result = vec3(0.0);

        if(enableDir) result += CalcDirLight(dirLight, Normal, viewDir);
        if(enablePoint) result += CalcPointLight(pointLight, Normal, FragPos, viewDir);
        if(enableSpot) result += CalcSpotLight(spotLight, Normal, FragPos, viewDir);

        LightingColor = result;
        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
)";

const std::string fragmentShaderGouraudSource = R"(
    #version 330 core
    out vec4 FragColor;

    in vec3 LightingColor;
    in vec2 TexCoords;

    uniform vec3 objectColor;
    uniform sampler2D diffuseMap;
    uniform int textureMode; 

    void main() {
        vec4 texColor;
        if(textureMode == 0) texColor = vec4(objectColor, 1.0);
        else if(textureMode == 1) texColor = texture(diffuseMap, TexCoords);
        else if(textureMode == 2) texColor = mix(vec4(objectColor, 1.0), texture(diffuseMap, TexCoords), 0.5);

        FragColor = vec4(LightingColor * texColor.rgb, texColor.a);
    }
)";

int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "GameHub: Shading & Texture Modes", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    Shader phongShader(vertexShaderPhongSource, fragmentShaderPhongSource);
    Shader gouraudShader(vertexShaderGouraudSource, fragmentShaderGouraudSource);

    unsigned int poolStickerTexture = loadTexture("resources/Designer.png");
    // Removed wall_image.jpg loading

    float vertices[] = {
        // positions          // normals           // texture coords
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 0.0f,

        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   0.0f, 0.0f,

        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f,

         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f,

        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f,

        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 1.0f
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

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        Shader& activeShader = useGouraud ? gouraudShader : phongShader;
        activeShader.use();

        activeShader.setBool("enableDir", dirLightOn);
        activeShader.setBool("enablePoint", pointLightOn);
        activeShader.setBool("enableSpot", spotLightOn);
        activeShader.setBool("enableAmbient", ambientOn);
        activeShader.setBool("enableDiffuse", diffuseOn);
        activeShader.setBool("enableSpecular", specularOn);
        activeShader.setVec3("viewPos", cameraPos);

        activeShader.setVec3("dirLight.direction", -0.2f, -1.0f, -0.3f);
        activeShader.setVec3("dirLight.ambient", 0.1f, 0.1f, 0.1f);
        activeShader.setVec3("dirLight.diffuse", 0.4f, 0.4f, 0.4f);
        activeShader.setVec3("dirLight.specular", 0.5f, 0.5f, 0.5f);

        activeShader.setVec3("pointLight.position", 0.0f, 5.8f, 0.0f);
        activeShader.setVec3("pointLight.ambient", 0.3f, 0.3f, 0.3f);
        activeShader.setVec3("pointLight.diffuse", 0.8f, 0.8f, 0.8f);
        activeShader.setVec3("pointLight.specular", 1.0f, 1.0f, 1.0f);
        activeShader.setFloat("pointLight.constant", 1.0f);
        activeShader.setFloat("pointLight.linear", 0.045f);
        activeShader.setFloat("pointLight.quadratic", 0.0075f);

        activeShader.setVec3("spotLight.position", -3.0f, 4.0f, 0.0f);
        activeShader.setVec3("spotLight.direction", 0.0f, -1.0f, 0.0f);
        activeShader.setVec3("spotLight.ambient", 0.0f, 0.0f, 0.0f);
        activeShader.setVec3("spotLight.diffuse", 1.0f, 0.9f, 0.7f);
        activeShader.setVec3("spotLight.specular", 1.0f, 1.0f, 1.0f);
        activeShader.setFloat("spotLight.constant", 1.0f);
        activeShader.setFloat("spotLight.linear", 0.09f);
        activeShader.setFloat("spotLight.quadratic", 0.032f);
        activeShader.setFloat("spotLight.cutOff", glm::cos(glm::radians(20.0f)));
        activeShader.setFloat("spotLight.outerCutOff", glm::cos(glm::radians(25.0f)));

        int displayW, displayH;
        glfwGetFramebufferSize(window, &displayW, &displayH);

        if (!isSplitView) {
            glViewport(0, 0, displayW, displayH);
            glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)displayW / (float)displayH, 0.1f, 100.0f);
            glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
            activeShader.setMat4("projection", projection);
            activeShader.setMat4("view", view);
            drawGameHubScene(VAO, activeShader, poolStickerTexture);
        }
        else {
            float ar = (float)(displayW / 2) / (float)(displayH / 2);
            glViewport(0, displayH / 2, displayW / 2, displayH / 2);
            glm::mat4 projection = glm::perspective(glm::radians(45.0f), ar, 0.1f, 100.0f);
            glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 15.0f, 0.1f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            activeShader.setMat4("projection", projection);  activeShader.setMat4("view", view);
            drawGameHubScene(VAO, activeShader, poolStickerTexture);

            glViewport(displayW / 2, displayH / 2, displayW / 2, displayH / 2);
            view = glm::lookAt(glm::vec3(12.0f, 3.0f, 0.0f), glm::vec3(0.0f, 3.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            activeShader.setMat4("view", view);
            drawGameHubScene(VAO, activeShader, poolStickerTexture);

            glViewport(0, 0, displayW / 2, displayH / 2);
            view = glm::lookAt(glm::vec3(0.0f, 2.0f, 12.0f), glm::vec3(0.0f, 2.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            activeShader.setMat4("view", view);
            drawGameHubScene(VAO, activeShader, poolStickerTexture);

            glViewport(displayW / 2, 0, displayW / 2, displayH / 2);
            view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
            activeShader.setMat4("view", view);
            drawGameHubScene(VAO, activeShader, poolStickerTexture);
        }
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glfwTerminate();
    return 0;
}

unsigned int loadTexture(char const* path) {
    unsigned int textureID;
    glGenTextures(1, &textureID);
    int width, height, nrComponents;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(path, &width, &height, &nrComponents, 0);
    if (data) {
        GLenum format;
        if (nrComponents == 1) format = GL_RED;
        else if (nrComponents == 3) format = GL_RGB;
        else if (nrComponents == 4) format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        stbi_image_free(data);
    }
    else {
        std::cout << "Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
    }
    return textureID;
}

void drawCube(unsigned int VAO, Shader& shader, glm::mat4 model, glm::vec3 color, unsigned int textureID) {
    shader.setMat4("model", model);
    shader.setVec3("objectColor", color);

    if (textureID > 0) {
        shader.setInt("textureMode", globalTextureMode);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);
        shader.setInt("diffuseMap", 0);
    }
    else {
        shader.setInt("textureMode", 0);
    }

    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
}

// ==========================================
// SCENE DRAWING
// ==========================================
void drawGameHubScene(unsigned int VAO, Shader& shader, unsigned int poolTexture) {
    glm::mat4 model;
    glm::vec3 wallColor = glm::vec3(0.85f, 0.9f, 0.9f);
    glm::vec3 floorColor = glm::vec3(0.4f, 0.4f, 0.4f);
    glm::vec3 roofColor = glm::vec3(0.9f, 0.9f, 0.9f);
    glm::vec3 glassColor = glm::vec3(0.4f, 0.7f, 0.9f);

    // 1. THE ROOM
    model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
    drawCube(VAO, shader, glm::scale(model, glm::vec3(16.0f, 0.1f, 12.0f)), floorColor);
    model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 6.0f, 0.0f));
    drawCube(VAO, shader, glm::scale(model, glm::vec3(16.0f, 0.1f, 12.0f)), roofColor);
    model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 3.0f, -6.0f));
    drawCube(VAO, shader, glm::scale(model, glm::vec3(16.0f, 6.0f, 0.1f)), wallColor);

    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-8.0f, 1.0f, 0.0f)), glm::vec3(0.1f, 2.0f, 12.0f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-8.0f, 5.0f, 0.0f)), glm::vec3(0.1f, 2.0f, 12.0f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-8.0f, 3.0f, -4.0f)), glm::vec3(0.1f, 2.0f, 4.0f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-8.0f, 3.0f, 4.0f)), glm::vec3(0.1f, 2.0f, 4.0f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-8.0f, 3.0f, 0.0f)), glm::vec3(0.05f, 2.0f, 4.0f)), glassColor);

    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(8.0f, 1.0f, 0.0f)), glm::vec3(0.1f, 2.0f, 12.0f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(8.0f, 5.0f, 0.0f)), glm::vec3(0.1f, 2.0f, 12.0f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(8.0f, 3.0f, -4.0f)), glm::vec3(0.1f, 2.0f, 4.0f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(8.0f, 3.0f, 4.0f)), glm::vec3(0.1f, 2.0f, 4.0f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(8.0f, 3.0f, 0.0f)), glm::vec3(0.05f, 2.0f, 4.0f)), glassColor);

    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(-4.5f, 3.0f, 6.0f)), glm::vec3(7.0f, 6.0f, 0.1f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(4.5f, 3.0f, 6.0f)), glm::vec3(7.0f, 6.0f, 0.1f)), wallColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 4.5f, 6.0f)), glm::vec3(2.0f, 3.0f, 0.1f)), wallColor);
    glm::mat4 door = glm::translate(glm::mat4(1.0f), glm::vec3(-1.0f, 1.5f, 6.0f));
    door = glm::rotate(door, glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    door = glm::translate(door, glm::vec3(1.0f, 0.0f, 0.0f));
    drawCube(VAO, shader, glm::scale(door, glm::vec3(2.0f, 3.0f, 0.05f)), glm::vec3(0.4f, 0.2f, 0.1f));

    // 2. POOL (BILLIARDS) TABLE
    glm::vec3 poolCenter = glm::vec3(-3.0f, 0.8f, 0.0f);
    glm::vec3 woodColor = glm::vec3(0.3f, 0.15f, 0.05f);
    glm::vec3 feltGreen = glm::vec3(0.1f, 0.5f, 0.2f);

    model = glm::translate(glm::mat4(1.0f), poolCenter);
    model = glm::scale(model, glm::vec3(3.0f, 0.1f, 4.5f));
    drawCube(VAO, shader, model, feltGreen, 0);

    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), poolCenter + glm::vec3(0.0f, 0.1f, -2.3f)), glm::vec3(3.2f, 0.2f, 0.2f)), woodColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), poolCenter + glm::vec3(0.0f, 0.1f, 2.3f)), glm::vec3(3.2f, 0.2f, 0.2f)), woodColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), poolCenter + glm::vec3(-1.6f, 0.1f, 0.0f)), glm::vec3(0.2f, 0.2f, 4.4f)), woodColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), poolCenter + glm::vec3(1.6f, 0.1f, 0.0f)), glm::vec3(0.2f, 0.2f, 4.4f)), woodColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), poolCenter + glm::vec3(-1.3f, -0.4f, -2.0f)), glm::vec3(0.2f, 0.8f, 0.2f)), woodColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), poolCenter + glm::vec3(1.3f, -0.4f, -2.0f)), glm::vec3(0.2f, 0.8f, 0.2f)), woodColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), poolCenter + glm::vec3(-1.3f, -0.4f, 2.0f)), glm::vec3(0.2f, 0.8f, 0.2f)), woodColor);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), poolCenter + glm::vec3(1.3f, -0.4f, 2.0f)), glm::vec3(0.2f, 0.8f, 0.2f)), woodColor);

    glm::mat4 cue = glm::translate(glm::mat4(1.0f), poolCenter + glm::vec3(0.0f, 0.4f, 1.2f));
    cue = glm::rotate(cue, glm::radians(-10.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    drawCube(VAO, shader, glm::scale(cue, glm::vec3(0.04f, 0.04f, 2.5f)), glm::vec3(0.8f, 0.7f, 0.5f));

    // 3. TABLE TENNIS
    glm::vec3 ttCenter = glm::vec3(4.0f, 0.8f, -2.0f);
    glm::vec3 blueTable = glm::vec3(0.1f, 0.3f, 0.7f);
    glm::vec3 metalLegs = glm::vec3(0.2f, 0.2f, 0.2f);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), ttCenter), glm::vec3(3.0f, 0.05f, 1.6f)), blueTable);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), ttCenter + glm::vec3(0.0f, 0.1f, 0.0f)), glm::vec3(0.02f, 0.15f, 1.7f)), glm::vec3(0.9f));
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), ttCenter + glm::vec3(-1.4f, -0.4f, -0.7f)), glm::vec3(0.05f, 0.8f, 0.05f)), metalLegs);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), ttCenter + glm::vec3(-1.4f, -0.4f, 0.7f)), glm::vec3(0.05f, 0.8f, 0.05f)), metalLegs);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), ttCenter + glm::vec3(1.4f, -0.4f, -0.7f)), glm::vec3(0.05f, 0.8f, 0.05f)), metalLegs);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), ttCenter + glm::vec3(1.4f, -0.4f, 0.7f)), glm::vec3(0.05f, 0.8f, 0.05f)), metalLegs);

    // 4. CARROM BOARD
    glm::vec3 carromCenter = glm::vec3(4.0f, 0.6f, 3.0f);
    glm::vec3 frameWood = glm::vec3(0.3f, 0.15f, 0.05f);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), carromCenter - glm::vec3(0.0f, 0.3f, 0.0f)), glm::vec3(0.8f, 0.6f, 0.8f)), metalLegs);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), carromCenter), glm::vec3(1.2f, 0.05f, 1.2f)), glm::vec3(0.8f, 0.6f, 0.4f));
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), carromCenter + glm::vec3(0.0f, 0.05f, -0.65f)), glm::vec3(1.4f, 0.1f, 0.1f)), frameWood);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), carromCenter + glm::vec3(0.0f, 0.05f, 0.65f)), glm::vec3(1.4f, 0.1f, 0.1f)), frameWood);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), carromCenter + glm::vec3(-0.65f, 0.05f, 0.0f)), glm::vec3(0.1f, 0.1f, 1.4f)), frameWood);
    drawCube(VAO, shader, glm::scale(glm::translate(glm::mat4(1.0f), carromCenter + glm::vec3(0.65f, 0.05f, 0.0f)), glm::vec3(0.1f, 0.1f, 1.4f)), frameWood);
}

// ==========================================
// INPUT CONTROLS
// ==========================================
void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    float cameraSpeed = 4.5f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) cameraPos -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;

    // --- Shading Model Controls ---
    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_G]) { useGouraud = true; cout << "Gouraud Shading ON" << endl; keyProcessed[GLFW_KEY_G] = true; }
    }
    else keyProcessed[GLFW_KEY_G] = false;

    if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_H]) { useGouraud = false; cout << "Phong Shading ON" << endl; keyProcessed[GLFW_KEY_H] = true; }
    }
    else keyProcessed[GLFW_KEY_H] = false;

    // --- Texture Mode Controls ---
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_Z]) { globalTextureMode = 0; cout << "Texture OFF (Color Only)" << endl; keyProcessed[GLFW_KEY_Z] = true; }
    }
    else keyProcessed[GLFW_KEY_Z] = false;

    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_X]) { globalTextureMode = 1; cout << "Texture ON" << endl; keyProcessed[GLFW_KEY_X] = true; }
    }
    else keyProcessed[GLFW_KEY_X] = false;

    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_C]) { globalTextureMode = 2; cout << "Texture BLEND Mode" << endl; keyProcessed[GLFW_KEY_C] = true; }
    }
    else keyProcessed[GLFW_KEY_C] = false;

    // --- Light Controls ---
    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_1]) { dirLightOn = !dirLightOn; keyProcessed[GLFW_KEY_1] = true; }
    }
    else keyProcessed[GLFW_KEY_1] = false;

    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_2]) { pointLightOn = !pointLightOn; keyProcessed[GLFW_KEY_2] = true; }
    }
    else keyProcessed[GLFW_KEY_2] = false;

    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_3]) { spotLightOn = !spotLightOn; keyProcessed[GLFW_KEY_3] = true; }
    }
    else keyProcessed[GLFW_KEY_3] = false;

    if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_5]) { ambientOn = !ambientOn; keyProcessed[GLFW_KEY_5] = true; }
    }
    else keyProcessed[GLFW_KEY_5] = false;

    if (glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_6]) { diffuseOn = !diffuseOn; keyProcessed[GLFW_KEY_6] = true; }
    }
    else keyProcessed[GLFW_KEY_6] = false;

    if (glfwGetKey(window, GLFW_KEY_7) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_7]) { specularOn = !specularOn; keyProcessed[GLFW_KEY_7] = true; }
    }
    else keyProcessed[GLFW_KEY_7] = false;

    // --- Viewport Control ---
    if (glfwGetKey(window, GLFW_KEY_9) == GLFW_PRESS) {
        if (!keyProcessed[GLFW_KEY_9]) { isSplitView = !isSplitView; keyProcessed[GLFW_KEY_9] = true; }
    }
    else keyProcessed[GLFW_KEY_9] = false;
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
    float xpos = static_cast<float>(xposIn); float ypos = static_cast<float>(yposIn);
    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
    float xoffset = xpos - lastX; float yoffset = lastY - ypos;
    lastX = xpos; lastY = ypos;
    cameraYaw += xoffset * 0.1f; cameraPitch += yoffset * 0.1f;
    if (cameraPitch > 89.0f) cameraPitch = 89.0f; if (cameraPitch < -89.0f) cameraPitch = -89.0f;
    glm::vec3 front;
    front.x = cos(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
    front.y = sin(glm::radians(cameraPitch));
    front.z = sin(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
    cameraFront = glm::normalize(front);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {}