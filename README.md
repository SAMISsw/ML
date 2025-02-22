# ML
```swift
import SwiftUI
import CoreML
import Foundation

class TermaiModelHandler {
    private var model: Termai
    private var userDefaultsKey = "UserMessagesData"
    
    private var trainingData: [(input: String, output: String)] = []
    
    init() {
        self.model = loadModel()
        self.trainingData = fetchStoredData()
    }
    
    private func loadModel() -> Termai {
        do {
            let config = MLModelConfiguration()
            let model = try Termai(configuration: config)
            return model
        } catch {
            fatalError("Erro ao carregar o modelo: \(error)")
        }
    }
    
    func retrainModel(newInput: String, newOutput: String) {
        trainingData.append((input: newInput, output: newOutput))
        saveDataToUserDefaults(trainingData)
    }
    
    private func fetchStoredData() -> [(input: String, output: String)] {
        if let storedData = UserDefaults.standard.array(forKey: userDefaultsKey) as? [[String]] {
            return storedData.map { ($0[0], $0[1]) }
        }
        return []
    }
    
    private func saveDataToUserDefaults(_ data: [(input: String, output: String)]) {
        let storedData = data.map { [$0.input, $0.output] }
        UserDefaults.standard.set(storedData, forKey: userDefaultsKey)
    }
    
    func generateResponse(for input: String) -> String {
        var response = "Erro ao gerar resposta."
        
        for item in trainingData {
            if item.input.lowercased().contains(input.lowercased()) {
                response = item.output
                break
            }
        }
        
        return response
    }
}

struct ContentView: View {
    @State private var userInput: String = ""
    @State private var messages: [String] = []
    
    private var modelHandler = TermaiModelHandler()

    var body: some View {
        VStack {
            ScrollView {
                VStack(alignment: .leading) {
                    ForEach(messages, id: \.self) { message in
                        Text(message)
                            .padding()
                            .background(Color.black.opacity(0.1))
                            .cornerRadius(10)
                            .padding(.vertical, 2)
                    }
                }
                .padding()
            }
            
            Spacer()
            
            HStack {
                TextField("Digite algo...", text: $userInput)
                    .padding(10)
                    .background(Color.black.opacity(0.1))
                    .cornerRadius(15)
                    .foregroundColor(.black)
                    .padding(.leading, 10)
                
                Button(action: sendMessage) {
                    Image(systemName: "magnifyingglass")
                        .padding()
                        .background(Color.blue)
                        .clipShape(Circle())
                        .foregroundColor(.white)
                }
                .padding(.trailing, 10)
            }
            .frame(height: 50)
            .background(Color.black.opacity(0.1))
            .cornerRadius(20)
            .padding()
            
            Button(action: copyText) {
                Image(systemName: "doc.on.clipboard")
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .clipShape(Circle())
            }
            .padding()
        }
        .background(Color.white)
        .edgesIgnoringSafeArea(.bottom)
    }
    
    func sendMessage() {
        let userMessage = userInput.trimmingCharacters(in: .whitespacesAndNewlines)
        if !userMessage.isEmpty {
            messages.append("Você: \(userMessage)")
            userInput = ""
            let aiResponse = generateAIResponse(for: userMessage)
            messages.append("Termai: \(aiResponse)")
            modelHandler.retrainModel(newInput: userMessage, newOutput: aiResponse)
        }
    }
    
    func generateAIResponse(for input: String) -> String {
        return modelHandler.generateResponse(for: input)
    }
    
    func copyText() {
        let responseText = messages.last ?? ""
        UIPasteboard.general.string = responseText
    }
}

```

## Train

```swift
import CreateML
import Foundation

func combineDatasets(csvPaths: [String]) -> MLDataTable? {
    var dataTables: [MLDataTable] = []
    
    for path in csvPaths {
        do {
            let dataTable = try MLDataTable(contentsOf: URL(fileURLWithPath: path))
            dataTables.append(dataTable)
        } catch {
            print("Erro ao carregar o dataset de \(path): \(error)")
            return nil
        }
    }
    
    return dataTables.reduce(nil) { (combinedDataTable, dataTable) in
        if let combined = combinedDataTable {
            return combined.appending(dataTable)
        } else {
            return dataTable
        }
    }
}

let datasetPaths = [
    "/caminho/para/dataset1.csv",
    "/caminho/para/dataset2.csv",
    "/caminho/para/dataset3.csv"
]

if let combinedData = combineDatasets(csvPaths: datasetPaths) {
    do {
        let model = try MLClassifier(trainingData: combinedData, targetColumn: "label")
        let modelURL = URL(fileURLWithPath: "/caminho/para/salvar/o/modelo.mlmodel")
        try model.write(to: modelURL)
        print("Modelo treinado e salvo com sucesso em \(modelURL)")
    } catch {
        print("Erro ao treinar o modelo: \(error)")
    }
} else {
    print("Erro ao combinar os datasets.")
}
```
