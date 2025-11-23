#Project Layout
code_conversion_workspace/
  src/
    backend/
      app.py                # Flask app, routes
      config.py             # settings, thresholds
      llm_client.py         # Azure OpenAI wrapper
      ast_extractor.py      # run JavaParser helper and load ast.json
      graph_builder.py      # class & dependency graph
      chunker.py            # method chunking / summarisation
      doc_generator.py      # documentation creation
      validator.py          # skeleton + description validation
      converter.py          # Spring Boot generator
      test_generator.py     # JUnit generator
      data/
        pre_processed_files.json   # docs + metrics per class
        ast.json                   # output of Java helper
        logs/                      # log files
    frontend/
      templates/
        layout.html
        index.html
        documentation.html
      static/
        main.css
        main.js
  java_parser_helper/
    pom.xml
    src/main/java/extractor/Extractor.java
  requirements.txt
  README.md

#requirements.txt
flask
openai
numpy
pydantic
requests

#src/backend/config.py

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_DEPLOYMENT_CHAT = os.getenv("AZURE_DEPLOYMENT_CHAT", "gpt-35-turbo")
AZURE_DEPLOYMENT_EMBED = os.getenv("AZURE_DEPLOYMENT_EMBED", "text-embedding-3-small")

DATA_DIR = BASE_DIR / "src" / "backend" / "data"
AST_JSON_PATH = DATA_DIR / "ast.json"
PREPROCESSED_JSON = DATA_DIR / "pre_processed_files.json"

MAX_METHOD_CHARS = 6000              # to keep under ~4k tokens after prompt overhead
DOC_AUTO_RETRY = 1                   # how many times to retry doc generation
DOC_VALIDATION_THRESHOLD = 0.9       # avg skeleton+accuracy+coverage

JAVA_HELPER_DIR = BASE_DIR / "java_parser_helper"
JAVA_AST_OUT = AST_JSON_PATH
LEGACY_REPO_ENV = "LEGACY_JAVA_REPO" # optional env var, else user passes repo

#src/backend/llm_client.py

from __future__ import annotations
import json
from typing import List, Dict, Any
from openai import OpenAI
from .config import (
    AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT,
    AZURE_DEPLOYMENT_CHAT, AZURE_DEPLOYMENT_EMBED
)

client = OpenAI(
    api_key=AZURE_OPENAI_KEY,
    base_url=AZURE_OPENAI_ENDPOINT
)

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = client.embeddings.create(
        model=AZURE_DEPLOYMENT_EMBED,
        input=texts
    )
    return [e.embedding for e in resp.data]

def chat_json(system_prompt: str, user_payload: Any, max_tokens: int = 1500) -> Dict[str, Any]:
    """Call chat completion and parse JSON response."""
    resp = client.chat.completions.create(
        deployment_id=AZURE_DEPLOYMENT_CHAT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload)}
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    text = resp.choices[0].message.content
    try:
        return json.loads(text)
    except Exception:
        # if model added extra text, try to locate json substring
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        raise

# java_parser_helper/pom.xml

<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>

  <groupId>com.example</groupId>
  <artifactId>javaparser-extractor</artifactId>
  <version>1.0-SNAPSHOT</version>

  <properties>
    <maven.compiler.source>11</maven.compiler.source>
    <maven.compiler.target>11</maven.compiler.target>
  </properties>

  <dependencies>
    <dependency>
      <groupId>com.github.javaparser</groupId>
      <artifactId>javaparser-core</artifactId>
      <version>3.25.4</version>
    </dependency>
    <dependency>
      <groupId>com.google.code.gson</groupId>
      <artifactId>gson</artifactId>
      <version>2.10.1</version>
    </dependency>
  </dependencies>

  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-assembly-plugin</artifactId>
        <version>3.4.2</version>
        <configuration>
          <descriptorRefs>
            <descriptorRef>jar-with-dependencies</descriptorRef>
          </descriptorRefs>
          <archive>
            <manifest>
              <mainClass>extractor.Extractor</mainClass>
            </manifest>
          </archive>
        </configuration>
        <executions>
          <execution>
            <id>make-assembly</id>
            <phase>package</phase>
            <goals>
              <goal>single</goal>
            </goals>
          </execution>
        </executions>
      </plugin>
    </plugins>
  </build>
</project>

#java_parser_helper/src/main/java/extractor/Extractor.java

package extractor;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.type.Type;
import com.github.javaparser.utils.SourceRoot;
import com.google.gson.GsonBuilder;

import java.io.Writer;
import java.nio.file.*;
import java.util.*;

public class Extractor {
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: Extractor <repo-root> <out-json>");
            System.exit(1);
        }
        Path repo = Paths.get(args[0]);
        Path out = Paths.get(args[1]);

        List<Map<String,Object>> files = new ArrayList<>();
        Files.walk(repo)
                .filter(p -> p.toString().endsWith(".java"))
                .forEach(p -> {
                    try {
                        String src = Files.readString(p);
                        CompilationUnit cu = JavaParser.parse(src);

                        Map<String,Object> fileObj = new LinkedHashMap<>();
                        fileObj.put("path", repo.relativize(p).toString());
                        fileObj.put("source", src);

                        List<Map<String,Object>> types = new ArrayList<>();
                        for (TypeDeclaration<?> td : cu.getTypes()) {
                            types.add(extractType(td));
                        }
                        fileObj.put("types", types);
                        files.add(fileObj);
                    } catch (Exception e) {
                        System.err.println("Failed parse "+p+": "+e.getMessage());
                    }
                });

        Map<String,Object> root = new LinkedHashMap<>();
        root.put("files", files);
        try (Writer w = Files.newBufferedWriter(out)) {
            w.write(new GsonBuilder().setPrettyPrinting().create().toJson(root));
        }
        System.out.println("Wrote AST JSON to " + out.toAbsolutePath());
    }

    private static Map<String,Object> extractType(TypeDeclaration<?> td) {
        Map<String,Object> typeObj = new LinkedHashMap<>();
        typeObj.put("name", td.getNameAsString());
        typeObj.put("kind", td.getClass().getSimpleName());
        typeObj.put("modifiers", td.getModifiers().toString());

        List<Map<String,Object>> fields = new ArrayList<>();
        for (FieldDeclaration fd : td.getFields()) {
            for (VariableDeclarator var : fd.getVariables()) {
                Map<String,Object> f = new LinkedHashMap<>();
                f.put("name", var.getNameAsString());
                f.put("type", var.getType().asString());
                f.put("modifiers", fd.getModifiers().toString());
                fields.add(f);
            }
        }
        typeObj.put("fields", fields);

        List<Map<String,Object>> methods = new ArrayList<>();
        for (MethodDeclaration md : td.getMethods()) {
            Map<String,Object> m = new LinkedHashMap<>();
            m.put("name", md.getNameAsString());
            m.put("returnType", md.getType().asString());
            m.put("modifiers", md.getModifiers().toString());
            m.put("beginLine", md.getBegin().isPresent() ? md.getBegin().get().line : -1);
            m.put("endLine", md.getEnd().isPresent() ? md.getEnd().get().line : -1);
            m.put("body", md.getBody().isPresent() ? md.getBody().get().toString() : "");
            List<Map<String,String>> params = new ArrayList<>();
            md.getParameters().forEach(p -> {
                Map<String,String> param = new LinkedHashMap<>();
                param.put("name", p.getNameAsString());
                param.put("type", p.getType().asString());
                params.add(param);
            });
            m.put("parameters", params);
            methods.add(m);
        }
        typeObj.put("methods", methods);

        List<Map<String,Object>> inner = new ArrayList<>();
        for (BodyDeclaration<?> bd : td.getMembers()) {
            if (bd.isClassOrInterfaceDeclaration()) {
                inner.add(extractType((TypeDeclaration<?>)bd));
            }
        }
        typeObj.put("innerClasses", inner);

        return typeObj;
    }
}

#Build the jar once
cd java_parser_helper
mvn -q clean package

#src/backend/ast_extractor.py

from __future__ import annotations
import json, subprocess, shutil
from pathlib import Path
from typing import Dict, Any
from .config import JAVA_HELPER_DIR, JAVA_AST_OUT

def run_java_extractor(repo_path: str) -> None:
    jar_candidates = list((JAVA_HELPER_DIR / "target").glob("*jar-with-dependencies.jar"))
    if not jar_candidates:
        raise RuntimeError("Java helper jar not found. Build with `mvn -q clean package` in java_parser_helper/")
    jar = jar_candidates[0]
    java = shutil.which("java")
    if not java:
        raise RuntimeError("java not found in PATH")

    out = str(JAVA_AST_OUT)
    cmd = [java, "-jar", str(jar), repo_path, out]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Extractor failed: {proc.stderr}")
    print(proc.stdout)

def load_ast() -> Dict[str, Any]:
    return json.loads(JAVA_AST_OUT.read_text(encoding="utf-8"))

#src/backend/chunker.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from .config import MAX_METHOD_CHARS
from .llm_client import chat_json

@dataclass
class MethodChunk:
    class_name: str
    method_name: str
    signature: Dict[str, Any]
    body_segments: List[str]
    summary: str | None = None

SEGMENT_SUMMARY_SYSTEM = """
You are summarising Java method segments.
Given {methodName} and a code segment, produce a concise description of what this segment does.
Return JSON: {"summary": "..."}.
"""

METHOD_SUMMARY_SYSTEM = """
You are summarising a Java method from multiple segment summaries.
Input: method signature and list of segment summaries.
Return JSON: {"description": "..."} with a short but precise description.
"""

def split_long_method(body: str) -> List[str]:
    if len(body) <= MAX_METHOD_CHARS:
        return [body]
    segments = []
    current = []
    count = 0
    for line in body.splitlines():
        current.append(line)
        count += len(line)
        if count >= MAX_METHOD_CHARS or line.strip().endswith(";") or line.strip().startswith(("if", "for", "while", "switch")):
            segments.append("\n".join(current))
            current, count = [], 0
    if current:
        segments.append("\n".join(current))
    return segments

def summarise_method(class_name: str, method_meta: Dict[str, Any]) -> MethodChunk:
    body = method_meta.get("body", "")
    segments = split_long_method(body)
    seg_summaries = []
    for seg in segments:
        payload = {"methodName": method_meta["name"], "segment": seg}
        j = chat_json(SEGMENT_SUMMARY_SYSTEM, payload, max_tokens=300)
        seg_summaries.append(j.get("summary", ""))

    payload2 = {
        "method": {
            "name": method_meta["name"],
            "returnType": method_meta.get("returnType"),
            "parameters": method_meta.get("parameters", [])
        },
        "segments": seg_summaries
    }
    j2 = chat_json(METHOD_SUMMARY_SYSTEM, payload2, max_tokens=300)
    return MethodChunk(
        class_name=class_name,
        method_name=method_meta["name"],
        signature=payload2["method"],
        body_segments=segments,
        summary=j2.get("description", "")
    )

#src/backend/doc_generator.py

from __future__ import annotations
import json
from typing import Dict, Any
from .llm_client import chat_json
from .chunker import summarise_method
from .config import DOC_AUTO_RETRY
from .validator import validate_doc_against_ast

DOC_SYSTEM_PROMPT = """
You are generating documentation for a single Java class.
Input JSON contains:
- "skeleton": AST-based structure (className, fields, methods, innerClasses).
- "methods": list of {name, description, parameters, returnType, ...} summarised from code.

Produce JSON documentation of this shape:
{
  "components": [
    {
      "name": "<className>",
      "description": "...",
      "fields": [{ "name": "...", "type": "...", "description": "..." }],
      "methods": [
        {
          "name": "...",
          "description": "...",
          "parameters": [{"name":"...","type":"...","description":"..."}],
          "return": {"type":"...", "description":"..."}
        }
      ]
    }
  ]
}
Be faithful to the skeleton; do not invent classes, fields or methods.
"""

def build_skeleton(type_obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "className": type_obj["name"],
        "kind": type_obj["kind"],
        "modifiers": type_obj.get("modifiers", ""),
        "fields": type_obj.get("fields", []),
        "methods": [
            {k: v for k, v in m.items() if k in ("name", "returnType", "parameters", "modifiers")}
            for m in type_obj.get("methods", [])
        ],
        "innerClasses": [build_skeleton(ic) for ic in type_obj.get("innerClasses", [])]
    }

def generate_documentation_for_class(type_obj: Dict[str, Any], max_retries: int | None = None) -> Dict[str, Any]:
    if max_retries is None:
        max_retries = DOC_AUTO_RETRY

    attempts = 0
    best_doc = None
    best_score = -1.0

    while attempts <= max_retries:
        attempts += 1
        # summarise methods
        method_docs = []
        for m in type_obj.get("methods", []):
            mc = summarise_method(type_obj["name"], m)
            method_docs.append({
                "name": mc.method_name,
                "description": mc.summary,
                "parameters": mc.signature.get("parameters", []),
                "returnType": mc.signature.get("returnType")
            })
        payload = {
            "skeleton": build_skeleton(type_obj),
            "methods": method_docs
        }
        doc_json = chat_json(DOC_SYSTEM_PROMPT, payload, max_tokens=1400)

        # run validation
        val = validate_doc_against_ast(type_obj, doc_json)
        score = (val["skeleton"]["total_completeness_percentage"]
                 + val["description"]["accuracy_percentage"]
                 + val["description"]["completeness_percentage"]) / 3.0

        if score > best_score:
            best_score = score
            best_doc = {"doc": doc_json, "validation": val}

        if score >= 100 * 0.999 or (score / 100.0) >= 0.9:
            break

    return best_doc

#src/backend/validator.py

from __future__ import annotations
from typing import Dict, Any, List
from .llm_client import chat_json

VALIDATOR_SYSTEM = """
You are validating class documentation for a Java class.
Input:
{
  "skeleton": {...},    // AST-based structure
  "doc": {...}          // generated documentation
}
Return JSON:
{
  "accuracy_percentage": 100.0,
  "completeness_percentage": 100.0,
  "explanation": "..."
}
Accuracy = how well descriptions match intended behavior.
Completeness = whether all fields/methods in skeleton are documented.
"""

def _skeleton_metrics(type_obj: Dict[str, Any], doc: Dict[str, Any]) -> Dict[str, Any]:
    """Pure structural checks, similar to your skeleton_validation screenshot."""
    ast_fields = {f["name"] for f in type_obj.get("fields", [])}
    ast_methods = {m["name"] for m in type_obj.get("methods", [])}

    doc_components = doc.get("components", [])
    doc_fields, doc_methods = set(), set()
    for comp in doc_components:
        for f in comp.get("fields", []):
            doc_fields.add(f.get("name"))
        for m in comp.get("methods", []):
            doc_methods.add(m.get("name"))

    total_fields = len(ast_fields) or 1
    total_methods = len(ast_methods) or 1

    fields_covered = len(ast_fields & doc_fields) / total_fields
    methods_covered = len(ast_methods & doc_methods) / total_methods

    total = (fields_covered + methods_covered) / 2.0

    return {
        "total_fields": len(ast_fields),
        "fields_completeness_score": fields_covered * 100,
        "total_methods": len(ast_methods),
        "methods_completeness_score": methods_covered * 100,
        "total_completeness_percentage": total * 100
    }

def validate_doc_against_ast(type_obj: Dict[str, Any], doc: Dict[str, Any]) -> Dict[str, Any]:
    skeleton = _skeleton_metrics(type_obj, doc)
    payload = {"skeleton": type_obj, "doc": doc}
    desc = chat_json(VALIDATOR_SYSTEM, payload, max_tokens=600)

    description = {
        "accuracy_percentage": desc.get("accuracy_percentage", 0.0),
        "completeness_percentage": desc.get("completeness_percentage", 0.0),
        "explanation": desc.get("explanation", "")
    }
    return {
        "skeleton_validation": skeleton,
        "description_validation": description
    }

#src/backend/converter.py

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import base64
from pathlib import Path
from .llm_client import chat_json

CONVERTER_SYSTEM = """
You are converting documentation of a Java service into Spring Boot Java code.
Input JSON is the documentation with components/methods.
Output JSON:
{
  "files":[{"path":"src/main/java/.../X.java","content":"<base64>"}],
  "pom": "<base64 pom.xml>",
  "notes": "..."
}
Use Java 11 and a recent Spring Boot version. Do not use Lombok.
"""

def convert_doc_to_spring(doc: Dict[str, Any]) -> Dict[str, Any]:
    j = chat_json(CONVERTER_SYSTEM, doc, max_tokens=2200)
    files: List[Tuple[str,str]] = []
    for f in j.get("files", []):
        path = f.get("path")
        try:
            content = base64.b64decode(f.get("content","")).decode("utf-8")
        except Exception:
            content = f.get("content","")
        files.append((path, content))

    pom_content = ""
    if "pom" in j:
        try:
            pom_content = base64.b64decode(j["pom"]).decode("utf-8")
        except Exception:
            pom_content = j["pom"]

    return {"files": files, "pom": pom_content, "notes": j.get("notes","")}

#src/backend/test_generator.py

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import base64
from .llm_client import chat_json

TEST_SYSTEM = """
You are generating JUnit 5 tests for a Spring Boot service based on documentation.
Return JSON: {"tests":[{"path":"src/test/java/.../XTest.java","content":"<base64>"}]}
"""

def generate_tests(doc: Dict[str, Any]) -> List[Tuple[str,str]]:
    j = chat_json(TEST_SYSTEM, doc, max_tokens=1600)
    tests: List[Tuple[str,str]] = []
    for t in j.get("tests", []):
        path = t.get("path")
        try:
            content = base64.b64decode(t.get("content","")).decode("utf-8")
        except Exception:
            content = t.get("content","")
        tests.append((path, content))
    return tests

#src/backend/app.py (Flask app & orchestration)

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
from flask import Flask, render_template, jsonify, request
from .config import DATA_DIR, AST_JSON_PATH, PREPROCESSED_JSON
from .ast_extractor import load_ast, run_java_extractor
from .doc_generator import generate_documentation_for_class
from .converter import convert_doc_to_spring
from .test_generator import generate_tests

app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")

DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_preprocessed() -> Dict[str,Any]:
    if PREPROCESSED_JSON.exists():
        return json.loads(PREPROCESSED_JSON.read_text(encoding="utf-8"))
    return {"generation_details": {}}

def save_preprocessed(data: Dict[str,Any]):
    PREPROCESSED_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")

@app.route("/")
def index():
    ast = load_ast()
    pre = load_preprocessed()
    rows = []
    for f in ast["files"]:
        for type_obj in f.get("types", []):
            cls = type_obj["name"]
            gen = pre["generation_details"].get(cls)
            status = "Generated" if gen else "Pending"
            rows.append({
                "file_name": f["path"],
                "class_name": cls,
                "package": "(default)",
                "doc_status": "Generated" if gen else "Generate",
                "overall_status": status
            })
    return render_template("index.html", rows=rows)

@app.route("/classes")
def list_classes():
    ast = load_ast()
    out = []
    for f in ast["files"]:
        for type_obj in f.get("types", []):
            out.append({
                "file": f["path"], "class": type_obj["name"]
            })
    return jsonify(out)

@app.route("/classes/<class_name>/doc", methods=["POST"])
def generate_doc(class_name):
    ast = load_ast()
    target = None
    for f in ast["files"]:
        for t in f.get("types", []):
            if t["name"] == class_name:
                target = t
                break
    if not target:
        return jsonify({"error": "class not found"}), 404

    result = generate_documentation_for_class(target)
    pre = load_preprocessed()
    pre["generation_details"][class_name] = result
    save_preprocessed(pre)
    return jsonify({"status":"ok", "validation": result["validation"]})

@app.route("/documentation/<class_name>")
def documentation_view(class_name):
    pre = load_preprocessed()
    details = pre["generation_details"].get(class_name)
    if not details:
        return "No documentation yet. Generate first.", 404
    return render_template("documentation.html",
                           class_name=class_name,
                           doc_json=json.dumps(details["doc"], indent=2),
                           validation=details["validation"])

@app.route("/classes/<class_name>/convert", methods=["POST"])
def convert_to_spring(class_name):
    pre = load_preprocessed()
    details = pre["generation_details"].get(class_name)
    if not details:
        return jsonify({"error": "no documentation"}), 400

    conv = convert_doc_to_spring(details["doc"])
    out_dir = DATA_DIR.parent.parent / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    # write pom and files
    if conv["pom"]:
        (out_dir / "pom.xml").write_text(conv["pom"], encoding="utf-8")
    for path, content in conv["files"]:
        p = out_dir / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    return jsonify({"status": "ok", "notes": conv["notes"]})

@app.route("/classes/<class_name>/tests", methods=["POST"])
def generate_tests_endpoint(class_name):
    pre = load_preprocessed()
    details = pre["generation_details"].get(class_name)
    if not details:
        return jsonify({"error": "no documentation"}), 400
    tests = generate_tests(details["doc"])
    out_dir = DATA_DIR.parent.parent / "generated"
    for path, content in tests:
        p = out_dir / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return jsonify({"status":"ok", "tests_written": len(tests)})

def bootstrap(repo_path: str):
    # Run Java extractor once
    if not AST_JSON_PATH.exists():
        run_java_extractor(repo_path)

if __name__ == "__main__":
    import os
    repo = os.getenv("LEGACY_JAVA_REPO")
    if not repo:
        print("Set LEGACY_JAVA_REPO to your legacy Java repo path")
        raise SystemExit(1)
    bootstrap(repo)
    app.run(host="0.0.0.0", port=5001, debug=True)

#Simple UI templates (minimal but matches screenshots)
#src/frontend/templates/layout.html

<!doctype html>
<html>
<head>
    <title>Code Conversion Tool</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='main.css') }}" />
</head>
<body>
    <div class="navbar">Code Conversion Tool</div>
    <div class="container">
        {% block content %}{% endblock %}
    </div>
</body>
</html>

#src/frontend/templates/index.html

{% extends "layout.html" %}
{% block content %}
<h2>Legacy Code Classes</h2>
<table>
  <thead>
    <tr>
      <th>File Name</th>
      <th>Class Name</th>
      <th>Package</th>
      <th>Documentation</th>
      <th>Status</th>
      <th>Actions</th>
    </tr>
  </thead>
  <tbody>
  {% for r in rows %}
    <tr>
      <td>{{ r.file_name }}</td>
      <td>{{ r.class_name }}</td>
      <td>{{ r.package }}</td>
      <td>
        {% if r.doc_status == "Generated" %}
            <span class="badge success">Generated</span>
        {% else %}
            <button onclick="generateDoc('{{ r.class_name }}')">Generate</button>
        {% endif %}
      </td>
      <td><span class="badge">{{ r.overall_status }}</span></td>
      <td><a href="{{ url_for('documentation_view', class_name=r.class_name) }}">Details</a></td>
    </tr>
  {% endfor %}
  </tbody>
</table>

<script>
async function generateDoc(className) {
  const res = await fetch(`/classes/${className}/doc`, {method: 'POST'});
  if (res.ok) location.reload();
  else alert('Generation failed');
}
</script>
{% endblock %}

#src/frontend/templates/documentation.html

{% extends "layout.html" %}
{% block content %}
<h2>Documentation: {{ class_name }}</h2>

<div class="toolbar">
  <button onclick="convertToSpring()">Convert to Spring</button>
  <button onclick="generateTests()">Generate Test Class</button>
  <span class="badge success">Auto-Validated</span>
</div>

<div class="split">
  <div class="left">
    <h3>Documentation</h3>
    <textarea style="width:100%;height:500px;">{{ doc_json }}</textarea>
  </div>
  <div class="right">
    <h3>Validation Metrics</h3>
    <h4>Skeleton Validation</h4>
    <pre>{{ validation.skeleton_validation | tojson(indent=2) }}</pre>
    <h4>Description Validation</h4>
    <pre>{{ validation.description_validation | tojson(indent=2) }}</pre>
  </div>
</div>

<script>
async function convertToSpring() {
  const res = await fetch(`/classes/{{ class_name }}/convert`, {method:'POST'});
  const j = await res.json();
  if (res.ok) alert('Spring code generated: ' + j.notes);
  else alert('Error: '+j.error);
}
async function generateTests() {
  const res = await fetch(`/classes/{{ class_name }}/tests`, {method:'POST'});
  const j = await res.json();
  if (res.ok) alert('Generated '+j.tests_written+' test files');
  else alert('Error: '+j.error);
}
</script>
{% endblock %}

#src/frontend/static/main.css

body { font-family: sans-serif; margin:0; }
.navbar { background:#222;color:#fff;padding:10px 20px;font-weight:bold; }
.container { padding:20px; }
table { border-collapse: collapse; width:100%; }
th, td { border:1px solid #ddd; padding:8px; }
th { background:#f5f5f5; }
.badge { padding:4px 8px; border-radius:4px; background:#ccc; }
.badge.success { background:#4caf50; color:#fff; }
button { padding:4px 10px; cursor:pointer; }
.split { display:flex; gap:20px; }
.left, .right { flex:1; }
textarea { font-family:monospace; font-size:12px; }

