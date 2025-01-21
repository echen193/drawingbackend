from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from typing import List
from pydantic import BaseModel

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
app = FastAPI()

class Node(BaseModel):
    id: str
    label: str
    x: float
    y: float
    selected: bool

class Edge(BaseModel):
    id: str
    from_: str
    to: str
    bondType: int
    selected: bool
    class Config:
        fields = {'from_': 'from'}
class NodeEdgeData(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
class SMILESInput(BaseModel):
    smiles: str

    
SCALE = 30
OFFSET = 5000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with your frontend's URL or use ["*"] for any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, OPTIONS, PUT, DELETE, etc.
    allow_headers=["*"],  # Allows all headers like Authorization, Content-Type, etc.
)
# Mols needed
# COC(=O)C1=CC=C(C2=CC=CC=C2)C=C1
# COC(=O)C1=CC(C2=CC=CC(Br)=C2)=CC=C1
# CC1=C(OC2=CC=CC(Br)=C2)C=CC=C1


# Function to return SMILES and draw molecule
@app.post("/get_smiles")
async def get_smiles(request: NodeEdgeData):
    # Access data directly from the Pydantic model
    nodes = request.nodes
    edges = request.edges

    # Log data to verify it
    print(f"Received Nodes: {nodes}")
    print(f"Received Edges: {edges}")

    # RDKit logic begins here:
    # Create an empty editable molecule
    editable_molecule = Chem.RWMol()

    # Add atoms (nodes)
    atom_indices = {}
    for node in nodes:
        atom_idx = editable_molecule.AddAtom(Chem.Atom(node.label))
        atom_indices[node.id] = atom_idx
    bond_type_mapping = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }
    chirality_mapping = {
        4: Chem.BondDir.BEGINWEDGE,
        5: Chem.BondDir.BEGINDASH,
    }
    # Edges logic:
    regular_bonds = [1,2,3]

    for edge in edges:
        start_idx = atom_indices[edge.from_]
        end_idx = atom_indices[edge.to]
        bond_type = edge.bondType
        if bond_type in regular_bonds:
            bond_type = bond_type_mapping[edge.bondType]
            editable_molecule.AddBond(start_idx, end_idx, bond_type)
        else:
            editable_molecule.AddBond(start_idx, end_idx, Chem.BondType.SINGLE)  # Add a default bond first
            bond = editable_molecule.GetBondBetweenAtoms(start_idx, end_idx)
            if bond:
                bond.SetBondDir(chirality_mapping[bond_type])

    # Convert to Mol object
    molecule = editable_molecule.GetMol()
    AllChem.Compute2DCoords(molecule)

    conformer = molecule.GetConformer()
    coordinates = []
    bonds = molecule.GetBonds()
    bondlist = []
    bondID = 0
    for bond in bonds:
        bondType = bond.GetBondType()
        begin = bond.GetBeginAtomIdx()  # Get the index of the starting atom
        end = bond.GetEndAtomIdx()
        bondDir = bond.GetBondDir()
        if bondDir == Chem.BondDir.BEGINWEDGE:
            bondType = 4
        elif bondDir == Chem.BondDir.BEGINDASH:
            bondType = 5
        bondlist.append({"bondType": bondType, "from": begin, "to": end, "id": bondID})  
        bondID += 1
    for atom in molecule.GetAtoms():
        pos = conformer.GetAtomPosition(atom.GetIdx())
        label = atom.GetSymbol()
        coordinates.append({'id': atom.GetIdx(), 'label': label, 'x': pos.x * SCALE, 'y': pos.y * SCALE})
    # Generate SMILES
    # Choosing not to sanitize to allow the user to draw whatever they want, even if the molecule is invalid.
    # Chem.SanitizeMol(molecule)
    smiles = Chem.MolToSmiles(molecule)



    img = Draw.MolToImage(molecule, size=(300, 300))
    img_path = 'molecule.png'
    img.save(img_path)
    Chem.MolToMolFile(molecule, 'mol.sdf')

    return {'smiles': smiles, 'coordinates' : coordinates, 'bonds': bondlist}



@app.post("/get_highlight")
async def get_highlight(request: NodeEdgeData):
    nodes = request.nodes
    edges = request.edges

    # Log data to verify it



    mol = Chem.MolFromMolFile("mol.sdf")
    if mol is None:
        return {"error": "Failed to load molecule"}
    selected_atoms = [int(node.id) for node in nodes if node.selected]

    selected_bonds = [
        bond.GetIdx()
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in selected_atoms and bond.GetEndAtomIdx() in selected_atoms]
    
    options = Draw.MolDrawOptions()
    options.includeAtomNumbers = True  
    options.highlightAtomLabels = {atom: str(atom) for atom in selected_atoms}
    img = Draw.MolToImage(
            mol,
            highlightAtoms=selected_atoms,
            highlightBonds=selected_bonds,
            highlightAtomColors={atom: (1.0, 0.5, 0.5) for atom in selected_atoms},  # Red for atoms
            size=(500, 500),
            options=options
        )
    img.save("highlighted_molecule.png")

@app.post("/process_smiles")
async def process_smiles(data: SMILESInput):
    try:
        smiles = data.smiles

        # Create RDKit molecule from SMILES
        molecule = Chem.MolFromSmiles(smiles)
        if not molecule:
            raise ValueError("Invalid SMILES string")
        Chem.Kekulize(molecule, clearAromaticFlags=True)
        # Generate 2D coordinates for rendering
        AllChem.Compute2DCoords(molecule)

        # Extract node and bond data
        conformer = molecule.GetConformer()
        coordinates = []
        bonds = []

        # Nodes (atoms)
        for atom in molecule.GetAtoms():
            pos = conformer.GetAtomPosition(atom.GetIdx())
            label = atom.GetSymbol()
            coordinates.append({
                "id": atom.GetIdx(),
                "label": label,
                "x": pos.x * SCALE,
                "y": pos.y * SCALE,
                "selected": False
            })

        # Bonds
        bondID = 0
        for bond in molecule.GetBonds():
            bondType = bond.GetBondType()
            bondTypeInt = {
                Chem.BondType.SINGLE: 1,
                Chem.BondType.DOUBLE: 2,
                Chem.BondType.TRIPLE: 3,
            }.get(bondType, 1)  # Default to single bond if type is unknown
            bonds.append({
                "id": bondID,
                "from": bond.GetBeginAtomIdx(),
                "to": bond.GetEndAtomIdx(),
                "bondType": bondTypeInt,
                "selected": False
            })
            bondID += 1

        # Regenerate SMILES for validation
        smiles_output = Chem.MolToSmiles(molecule)

        # Optional: Generate and save an image (if needed)
        img = Draw.MolToImage(molecule, size=(300, 300))
        img_path = 'molecule_from_smiles.png'
        img.save(img_path)
        Chem.MolToMolFile(molecule, 'mol.sdf')
        print(bonds)
        return {
            "smiles": smiles_output,
            "coordinates": coordinates,
            "bonds": bonds,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing SMILES: {str(e)}")
    
FILE_PATH = "smiles_data.json"

# Create JSON
if not os.path.exists(FILE_PATH):
    with open(FILE_PATH, "w") as file:
        json.dump([], file) 
@app.post("/save_smiles/")
async def save_smiles(input_data: SMILESInput):
    try:
        with open(FILE_PATH, "r") as f:
            data = json.load(f)
        if data:
            next_id = max(item["id"] for item in data) + 1
        else:
            next_id = 0

        # Create the new entry
        smiles_entry = {"id": next_id, "smiles": input_data.smiles}

        # Append the new entry to the data
        data.append(smiles_entry)

        # Save the updated data back to the file
        with open(FILE_PATH, "w") as file:
            json.dump(data, file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/compounds", response_model=List[dict])
def get_compounds():
    if not FILE_PATH:
        return []  # Return an empty list if the file doesn't exist
    with open(FILE_PATH, "r") as file:
        compounds = json.load(file)
    return compounds

def load_smiles_data():
    if FILE_PATH:
        with open(FILE_PATH, "r") as file:
            return json.load(file)
    return []
def save_smiles_data(data):
    with open(FILE_PATH, "w") as file:
        json.dump(data, file)

class DeleteRequest(BaseModel):
    ids: List[int]

@app.delete("/delete_smiles")
def delete_smiles(request: DeleteRequest):
    # Load the current data
    smiles_data = load_smiles_data()

    # Filter out items with IDs in the request
    ids_to_delete = set(request.ids)
    updated_smiles_data = [item for item in smiles_data if item["id"] not in ids_to_delete]

    # Save the updated data back to the file
    save_smiles_data(updated_smiles_data)

    deleted_count = len(smiles_data) - len(updated_smiles_data)
    return {
        "message": f"{deleted_count} item(s) successfully deleted",
        "deleted_ids": list(ids_to_delete - {item["id"] for item in updated_smiles_data}),
    }

"""
@app.put("/items/{item_id}")
async def create_item(item_id: int, item: Item):
    return {"item_id": item_id, **item.dict()}
"""
