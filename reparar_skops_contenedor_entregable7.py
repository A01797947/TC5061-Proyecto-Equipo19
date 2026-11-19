"""Compatibilidad skops entre Windows y contenedor (MiBici Entregable 7).

Ejecuta DESDE LA RAIZ DEL REPOSITORIO, con modelos propios verificados:
    python reparar_skops_contenedor_entregable7.py --confirmo-origen

En Windows skops puede clasificar TreePredictor como seguro y get_untrusted_types()
no lo informa; en skops 0.16 dentro del contenedor requiere permitirlo expresamente.
Autoriza UNICAMENTE este tipo en MLmodel para P2/P3 (modelos HGB del equipo),
tras comprobar SHA256 y sin modificar el binario model.skops.
"""
import argparse
import json
from pathlib import Path

import yaml
from deployment.api.inference import folder_digest

TRUST = 'sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor'
ROOT = Path(__file__).resolve().parent / 'deployment' / 'model_store'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmo-origen', action='store_true', help='Confirmo que P2/P3 fueron entrenados por mi equipo y que confío en su origen')
    args = parser.parse_args()
    if not args.confirmo_origen:
        parser.error('Confirma primero la procedencia segura con --confirmo-origen')

    mf = ROOT / 'manifest.json'
    saved_manifest = mf.read_bytes()
    manifest = json.loads(saved_manifest)
    originals = {}
    success = False
    try:
        # Evita editar archivos que no son los exportados y verificados.
        for key in ('p1', 'p2', 'p3'):
            expected = manifest['models'][key]['sha256']
            actual = folder_digest(ROOT / key)
            if expected != actual:
                raise RuntimeError(f'{key.upper()}: hash incorrecto ANTES de modificar: {actual} != {expected}')

        for key in ('p2', 'p3'):
            folder = ROOT / key
            mlmodel = folder / 'MLmodel'
            originals[mlmodel] = mlmodel.read_bytes()
            config = yaml.safe_load(originals[mlmodel])
            flavor = config['flavors']['sklearn']
            if flavor.get('serialization_format') != 'skops':
                raise RuntimeError(f'{key.upper()}: no se esperaba un modelo skops')
            allowed = list(flavor.get('skops_trusted_types') or [])
            if TRUST in allowed:
                print(f'{key.upper()}: TreePredictor ya autorizado')
                continue
            allowed.append(TRUST)
            flavor['skops_trusted_types'] = allowed
            mlmodel.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
            manifest['models'][key]['sha256'] = folder_digest(folder)
            print(f'{key.upper()}: se autorizó solamente TreePredictor y se recalculó SHA256')

        mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        from deployment.verify_snapshot import main as verify
        verify()
        success = True
    finally:
        if not success:
            for path, content in originals.items():
                path.write_bytes(content)
            mf.write_bytes(saved_manifest)
            print('Se restauraron los archivos originales debido al error.')

    print('OK: metadatos actualizados; ahora reconstruye la imagen SIN CACHE y prueba la API.')


if __name__ == '__main__':
    main()
