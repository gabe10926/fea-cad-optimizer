import sys
sys.path.append("/usr/lib/freecad/lib")

import FreeCAD as App
import Fem 

class CADController:

    def __init(self, model_path):
        self.doc = App.open(model_path)
        self.sheet = self.doc.Spreadsheet

    def set_parameter(self, cell, value):
        self.sheet.set(cell, str(value))
        self.doc.recompute()

    def export_inp(self, filepath):
        analysis = self.doc.Analysis
        Fem.exportAnalysis(analysis, filepath)

    def get_volume(self):
        total_volume = 0
        for obj in self.doc.Objects:
            if hasattr(obj, "Shape"):
                total_volume += obj.Shape.Volume
        return total_volume
    

                




