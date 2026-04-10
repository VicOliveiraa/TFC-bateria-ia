#include <Arduino.h>
#include <Wire.h>

#include <Adafruit_ADS1X15.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <INA226.h>

#include "FS.h"
#include "LittleFS.h"

#include <WiFi.h>
#include <WebServer.h>

// ================= PINOS =================
static const int PIN_I2C_SDA     = 21;
static const int PIN_I2C_SCL     = 22;
static const int PIN_TEMP_DQ     = 4;
static const int PIN_DISCH_CTRL  = 23;

// ================= I2C =================
static const uint32_t I2C_FREQ = 400000;

// ================= BAT_DIV (ADS) =================
// divisor: 4k7 (topo) e 10k (baixo) -> fator 1.47
static const float DIV_R1 = 4700.0f;
static const float DIV_R2 = 10000.0f;
static inline float divFactor() { return (DIV_R1 + DIV_R2) / DIV_R2; } // 1.47

// ================= INA226 =================
static const uint8_t INA_ADDR = 0x40;

// Se você está usando o módulo WCMCU com shunt onboard "R100" = 0.10 ohm:
static const float SHUNT_OHMS = 0.10f;

// INA226 regs
static const uint8_t REG_CONFIG = 0x00;
static const uint8_t REG_SHUNT  = 0x01;
static const uint8_t REG_BUS    = 0x02;

// LSBs (datasheet)
static const float INA_SHUNT_LSB_V = 2.5e-6f;  // 2.5 µV/bit
static const float INA_BUS_LSB_V   = 1.25e-3f; // 1.25 mV/bit

// ================= Temporização =================
static const uint32_t SAMPLE_MS  = 250;   // 4 Hz (monitor/atualização)
static const uint32_t LOG_MS     = 1000;  // 1 Hz (log)
static uint32_t tSample = 0;
static uint32_t tLog    = 0;

// ================= Temperatura (DS18B20) =================
static const uint32_t DS_CONV_MS = 750;
static uint32_t tDsRequest = 0;
static bool dsPending = false;

// ================= Logger (LittleFS) =================
static const char* LOG_PATH = "/log.csv";
static bool loggingEnabled  = false;
static bool fsOK            = false;

// ================= Monitor =================
static bool monitorEnabled = true;

// ================= Wi-Fi AP + WebServer =================
static const char* WIFI_SSID = "ESP32-LOGGER";
static const char* WIFI_PASS = "12345678";  // mínimo 8 caracteres
static bool webStarted = false;
WebServer server(80);

// ================= Objetos =================
Adafruit_ADS1115 ads;
INA226 ina(INA_ADDR, &Wire); // só begin()

OneWire oneWire(PIN_TEMP_DQ);
DallasTemperature dallas(&oneWire);

// ================= Estado =================
static bool dischargeEnabled = false;

static float lastV_ads    = NAN;  // Vbat via ADS (re-escalada)
static float lastV_bus    = NAN;  // Vbus via INA
static float lastV_sh     = NAN;  // Vsh (V, com sinal)
static float lastI_signed = NAN;  // A (com sinal)
static float lastT        = NAN;  // C
static float lastV_bat_mean = NAN;

static int sensePolarity = +1; // "pol invert" se CHG/DISCH invertidos
static const float I_IDLE_THRESH_A = 0.03f;

// ===================== I2C RAW INA226 =====================
static bool inaWriteReg16(uint8_t reg, uint16_t value) {
  Wire.beginTransmission(INA_ADDR);
  Wire.write(reg);
  Wire.write((uint8_t)(value >> 8));
  Wire.write((uint8_t)(value & 0xFF));
  return (Wire.endTransmission() == 0);
}

static bool inaReadReg16(uint8_t reg, uint16_t &out) {
  Wire.beginTransmission(INA_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)INA_ADDR, 2) != 2) return false;
  uint8_t msb = Wire.read();
  uint8_t lsb = Wire.read();
  out = ((uint16_t)msb << 8) | lsb;
  return true;
}

static void inaForceConfig() {
  // 0x4127 = AVG=1, VBUSCT=1.1ms, VSHCT=1.1ms, MODE=Shunt+Bus continuous
  uint16_t cfg = 0x4127;
  bool ok = inaWriteReg16(REG_CONFIG, cfg);
  Serial.printf("M, INA cfg=0x%04X -> %s\n", cfg, ok ? "OK" : "FAIL");
}

static bool readINA_raw(float &vbus_out, float &vsh_out, float &i_signed_out) {
  uint16_t shU = 0, busU = 0;
  if (!inaReadReg16(REG_SHUNT, shU)) return false;
  if (!inaReadReg16(REG_BUS, busU)) return false;

  int16_t rawSh = (int16_t)shU;
  uint16_t rawBus = busU;

  vsh_out  = (float)rawSh * INA_SHUNT_LSB_V; // V (com sinal)
  vbus_out = (float)rawBus * INA_BUS_LSB_V;  // V

  float i = vsh_out / SHUNT_OHMS;            // A (com sinal)
  i_signed_out = i * (float)sensePolarity;
  return true;
}

// ===================== ADS =====================
static bool readADS_Vbat(float &vbat_out) {
  int16_t raw = ads.readADC_SingleEnded(0);
  float vpin  = ads.computeVolts(raw);
  vbat_out = vpin * divFactor();
  return true;
}

// ===================== DS18B20 async =====================
static void dsKick() {
  dallas.requestTemperatures();
  tDsRequest = millis();
  dsPending = true;
}
static bool dsTry(float &tOut) {
  if (!dsPending) return false;
  if (millis() - tDsRequest < DS_CONV_MS) return false;
  float t = dallas.getTempCByIndex(0);
  dsPending = false;
  if (t == DEVICE_DISCONNECTED_C || t < -100.0f || t > 150.0f) return false;
  tOut = t;
  return true;
}

// ===================== Derivados =====================
static float computeVbatMean(float vbus, float vads) {
  const bool okBus = isfinite(vbus);
  const bool okAds = isfinite(vads);
  if (okBus && okAds) return 0.5f * (vbus + vads);
  if (okBus) return vbus;
  if (okAds) return vads;
  return NAN;
}

static const char* modeTextFromSignedI(float i_signed) {
  if (!isfinite(i_signed)) return "FAULT";
  if (fabs(i_signed) < I_IDLE_THRESH_A) return "IDLE";
  // Convenção: I>0 => DISCH, I<0 => CHG (você pode inverter com "pol invert")
  return (i_signed > 0) ? "DISCH" : "CHG";
}

static float currentAbs(float i_signed) {
  return isfinite(i_signed) ? fabs(i_signed) : NAN;
}

// ===================== Descarga =====================
static void setDischarge(bool en) {
  dischargeEnabled = en;
  digitalWrite(PIN_DISCH_CTRL, en ? HIGH : LOW);
  Serial.printf("E,%lu,DISCH,%s\n", (unsigned long)millis(), en ? "ON" : "OFF");
}

// ===================== LittleFS DIAG =====================
static void fsInfo() {
  Serial.println("M, --- LittleFS LIST ---");
  if (!fsOK) {
    Serial.println("M, LittleFS NOT mounted (fsOK=FALSE)");
    return;
  }
  File root = LittleFS.open("/");
  if (!root) { Serial.println("M, ERR: cannot open root"); return; }
  File file = root.openNextFile();
  if (!file) Serial.println("M, (root is empty)");
  while (file) {
    Serial.printf("M, file: %s  size:%u\n", file.name(), (unsigned)file.size());
    file = root.openNextFile();
  }
  Serial.println("M, ----------------------");
}

static void fsWriteTest() {
  Serial.println("M, fs test: creating /test.txt ...");
  if (!fsOK) { Serial.println("M, ERR: fsOK=FALSE"); return; }
  File f = LittleFS.open("/test.txt", FILE_WRITE);
  if (!f) { Serial.println("M, ERR: cannot FILE_WRITE /test.txt"); return; }
  f.println("ok");
  f.close();
  Serial.println("M, fs test: wrote /test.txt");
  fsInfo();
}

// ===================== LOG =====================
static bool logEnsureHeader() {
  if (!fsOK) {
    Serial.println("M, ERR: LittleFS not mounted -> cannot log");
    return false;
  }

  if (LittleFS.exists(LOG_PATH)) {
    File r = LittleFS.open(LOG_PATH, FILE_READ);
    if (r) {
      bool ok = (r.size() > 0);
      r.close();
      if (ok) return true;
    }
  }

  File w = LittleFS.open(LOG_PATH, FILE_WRITE);
  if (!w) {
    Serial.println("M, ERR: cannot FILE_WRITE /log.csv");
    return false;
  }

  w.println("t_ms,Vbat_mean,Vbus_INA,Vbat_ADS,Vsh_mV,I_A,mode,T_C");
  w.close();

  if (!LittleFS.exists(LOG_PATH)) {
    Serial.println("M, ERR: file still missing after header write");
    return false;
  }
  return true;
}

static void logNew() {
  Serial.println("M, log new: reset /log.csv");
  if (!fsOK) { Serial.println("M, ERR: fsOK=FALSE"); loggingEnabled = false; return; }

  LittleFS.remove(LOG_PATH);

  if (logEnsureHeader()) Serial.println("M, log ready (file created)");
  else { Serial.println("M, log FAILED"); loggingEnabled = false; }
}

static void logAppend(uint32_t ms) {
  if (!logEnsureHeader()) return;

  File f = LittleFS.open(LOG_PATH, FILE_APPEND);
  if (!f) { Serial.println("M, ERR: cannot FILE_APPEND /log.csv"); return; }

  const char* mode = modeTextFromSignedI(lastI_signed);
  float Iabs = currentAbs(lastI_signed);
  float Vsh_mV = isfinite(lastV_sh) ? (lastV_sh * 1000.0f) : NAN;

  char line[220];
  snprintf(line, sizeof(line),
           "%lu,%.4f,%.4f,%.4f,%.3f,%.4f,%s,%.2f",
           (unsigned long)ms,
           lastV_bat_mean,
           lastV_bus,
           lastV_ads,
           Vsh_mV,
           Iabs,
           mode,
           lastT);

  f.println(line);
  f.close();
}

static void logStatus() {
  if (!fsOK) { Serial.println("M, fsOK=FALSE"); return; }
  Serial.printf("M, logging=%s path=%s exists=%s\n",
                loggingEnabled ? "ON" : "OFF",
                LOG_PATH,
                LittleFS.exists(LOG_PATH) ? "YES" : "NO");
  if (LittleFS.exists(LOG_PATH)) {
    File f = LittleFS.open(LOG_PATH, FILE_READ);
    if (f) { Serial.printf("M, size=%u bytes\n", (unsigned)f.size()); f.close(); }
  }
}

// ===================== Monitor =====================
static void printMonitorLine() {
  const char* mode = modeTextFromSignedI(lastI_signed);
  float Iabs = currentAbs(lastI_signed);
  float Vsh_mV = isfinite(lastV_sh) ? (lastV_sh * 1000.0f) : NAN;

  Serial.printf(
    "\rMODE:%-5s | I:%6.3fA | Vsh:%7.3fmV | Vbat(mean):%5.3fV | Vbus(INA):%5.3fV | Vbat(ADS):%5.3fV | T:%5.2fC | LOG:%s   ",
    mode, Iabs, Vsh_mV,
    lastV_bat_mean, lastV_bus, lastV_ads, lastT,
    loggingEnabled ? "ON" : "OFF"
  );
}

// ===================== Web server (download) =====================
static void startWebDownloadServer() {
  if (!fsOK) {
    Serial.println("M, Web: fsOK=FALSE, cannot start web server");
    return;
  }

  WiFi.mode(WIFI_AP);
  WiFi.softAP(WIFI_SSID, WIFI_PASS);
  IPAddress ip = WiFi.softAPIP();

  server.on("/", HTTP_GET, []() {
    String html;
    html += "<html><body>";
    html += "<h2>ESP32 Logger</h2>";
    html += "<p><a href=\"/log.csv\">Download log.csv</a></p>";
    html += "<p><a href=\"/info\">Info</a></p>";
    html += "</body></html>";
    server.send(200, "text/html", html);
  });

  server.on("/info", HTTP_GET, []() {
    String s;
    s += "fsOK="; s += (fsOK ? "TRUE\n" : "FALSE\n");
    s += "exists(/log.csv)="; s += (LittleFS.exists("/log.csv") ? "YES\n" : "NO\n");
    if (LittleFS.exists("/log.csv")) {
      File f = LittleFS.open("/log.csv", FILE_READ);
      if (f) { s += "size=" + String((unsigned)f.size()) + " bytes\n"; f.close(); }
    }
    server.send(200, "text/plain", s);
  });

  // streaming do arquivo (serve pra logs grandes sem estourar RAM)
  server.on("/log.csv", HTTP_GET, []() {
    if (!LittleFS.exists("/log.csv")) {
      server.send(404, "text/plain", "log.csv not found");
      return;
    }
    File f = LittleFS.open("/log.csv", FILE_READ);
    if (!f) {
      server.send(500, "text/plain", "cannot open log.csv");
      return;
    }
    server.streamFile(f, "text/csv");
    f.close();
  });

  server.begin();
  webStarted = true;

  Serial.printf("\nM, WiFi AP ON: SSID=%s PASS=%s\n", WIFI_SSID, WIFI_PASS);
  Serial.printf("M, Acesse no PC/cel: http://%s/\n", ip.toString().c_str());
}

// ===================== Serial commands =====================
static void handleSerial() {
  static String buf;

  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\r') continue;

    if (c == '\n') {
      String cmd = buf;
      buf = "";
      cmd.trim();
      if (!cmd.length()) return;

      Serial.print("\n");

      if (cmd.equalsIgnoreCase("help")) {
        Serial.println("Commands:");
        Serial.println("  help");
        Serial.println("  status");
        Serial.println("  monitor on|off");
        Serial.println("  disch on|off");
        Serial.println("  pol invert          (inverte CHG/DISCH)");
        Serial.println("  ina cfg");
        Serial.println("  log new");
        Serial.println("  log start");
        Serial.println("  log stop");
        Serial.println("  log status");
        Serial.println("  fs info");
        Serial.println("  fs test");
        Serial.println("  wifi info");
        Serial.println();
        Serial.println("Download do log (arquivo grande):");
        Serial.println("  1) Conecte no Wi-Fi: ESP32-LOGGER (senha 12345678)");
        Serial.println("  2) Abra: http://192.168.4.1/log.csv");
        return;
      }

      if (cmd.equalsIgnoreCase("status")) {
        const char* mode = modeTextFromSignedI(lastI_signed);
        float Iabs = currentAbs(lastI_signed);
        float Vsh_mV = isfinite(lastV_sh) ? (lastV_sh * 1000.0f) : NAN;

        Serial.printf("M, MODE=%s I=%.4fA Vsh=%.3fmV Vbat_mean=%.4f Vbus=%.4f Vads=%.4f T=%.2fC disch=%d log=%d pol=%d fsOK=%d web=%d\n",
                      mode, Iabs, Vsh_mV,
                      lastV_bat_mean, lastV_bus, lastV_ads,
                      lastT,
                      dischargeEnabled ? 1 : 0,
                      loggingEnabled ? 1 : 0,
                      sensePolarity,
                      fsOK ? 1 : 0,
                      webStarted ? 1 : 0);
        return;
      }

      if (cmd.equalsIgnoreCase("monitor on"))  { monitorEnabled = true;  Serial.println("M, monitor=ON"); return; }
      if (cmd.equalsIgnoreCase("monitor off")) { monitorEnabled = false; Serial.println("M, monitor=OFF"); Serial.print("\n"); return; }

      if (cmd.equalsIgnoreCase("disch on"))  { setDischarge(true);  return; }
      if (cmd.equalsIgnoreCase("disch off")) { setDischarge(false); return; }

      if (cmd.equalsIgnoreCase("pol invert")) {
        sensePolarity *= -1;
        Serial.printf("M, sensePolarity=%d (I>0 => %s)\n",
                      sensePolarity,
                      (sensePolarity > 0) ? "DISCH" : "CHG");
        return;
      }

      if (cmd.equalsIgnoreCase("ina cfg")) { inaForceConfig(); return; }

      if (cmd.equalsIgnoreCase("log new"))   { logNew(); return; }
      if (cmd.equalsIgnoreCase("log start")) {
        if (!fsOK) { Serial.println("M, ERR: fsOK=FALSE -> cannot start logging"); loggingEnabled = false; }
        else { loggingEnabled = true; Serial.println("M, logging=ON"); }
        return;
      }
      if (cmd.equalsIgnoreCase("log stop"))  { loggingEnabled = false; Serial.println("M, logging=OFF"); return; }
      if (cmd.equalsIgnoreCase("log status")){ logStatus(); return; }

      if (cmd.equalsIgnoreCase("fs info"))   { fsInfo(); return; }
      if (cmd.equalsIgnoreCase("fs test"))   { fsWriteTest(); return; }

      if (cmd.equalsIgnoreCase("wifi info")) {
        if (!webStarted) { Serial.println("M, WiFi AP not started"); return; }
        Serial.printf("M, SSID=%s PASS=%s IP=%s\n", WIFI_SSID, WIFI_PASS, WiFi.softAPIP().toString().c_str());
        return;
      }

      Serial.println("M, unknown command. Type 'help'.");
      return;
    }

    if (c == 0x08 || c == 0x7F) {
      if (buf.length() > 0) buf.remove(buf.length() - 1);
      continue;
    }

    if (isPrintable(c)) buf += c;
  }
}

// ===================== Setup/Loop =====================
void setup() {
  Serial.begin(115200);
  delay(250);

  pinMode(PIN_DISCH_CTRL, OUTPUT);
  digitalWrite(PIN_DISCH_CTRL, LOW);

  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  Wire.setClock(I2C_FREQ);

  Serial.println("\nM, boot");

  // LittleFS (formata se necessário)
  fsOK = LittleFS.begin(true);
  Serial.printf("M, LittleFS.begin(true) = %s\n", fsOK ? "OK" : "FAIL");
  if (fsOK) fsInfo();

  // ADS
  if (!ads.begin()) {
    Serial.println("M, ADS1115 init FAILED");
  } else {
    ads.setGain(GAIN_ONE);
    Serial.println("M, ADS1115 ok (GAIN_ONE)");
  }

  // INA
  if (!ina.begin()) {
    Serial.println("M, INA226 init FAILED (lib begin)");
  } else {
    Serial.println("M, INA226 ok (lib begin)");
  }
  inaForceConfig();

  // Temp
  dallas.begin();
  dallas.setWaitForConversion(false);
  dsKick();

  setDischarge(false);

  // Web download (AP)
  startWebDownloadServer();

  Serial.println("M, ready. Type 'help'");
  tSample = millis();
  tLog    = millis();
}

void loop() {
  handleSerial();

  if (webStarted) server.handleClient();

  const uint32_t now = millis();

  // Atualiza leituras internas (4 Hz)
  if (now - tSample >= SAMPLE_MS) {
    tSample += SAMPLE_MS;

    float vAds = NAN;
    float vBus = NAN, vSh = NAN, iSigned = NAN;

    (void)readADS_Vbat(vAds);
    (void)readINA_raw(vBus, vSh, iSigned);

    float tC = NAN;
    if (dsTry(tC)) { lastT = tC; dsKick(); }

    lastV_ads    = vAds;
    lastV_bus    = vBus;
    lastV_sh     = vSh;
    lastI_signed = iSigned;

    lastV_bat_mean = computeVbatMean(lastV_bus, lastV_ads);

    if (monitorEnabled) printMonitorLine();
  }

  // Log 1 Hz (arquivo)
  if (loggingEnabled && (now - tLog >= LOG_MS)) {
    tLog += LOG_MS;
    logAppend(now);
  }
}
