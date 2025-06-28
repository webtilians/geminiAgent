# crear_db.py
import sqlite3
import os

# Nombre del archivo de la base de datos
DB_FILE = "rosaleda_camping.db"

def crear_base_de_datos():
    """
    Crea y puebla la base de datos desde cero.
    Si el archivo de la base de datos ya existe, lo borra para empezar de nuevo.
    """
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print(f"Antigua base de datos '{DB_FILE}' eliminada.")

    # Conectar a la base de datos (se creará si no existe)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # --- Definir la nueva estructura de tablas ---

    # Tabla de Alojamientos: Información general de cada tipo
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alojamientos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT NOT NULL UNIQUE,
        capacidad_max_personas INTEGER NOT NULL,
        mascotas_permitidas BOOLEAN NOT NULL DEFAULT 0,
        descripcion TEXT
    );
    """)

    # Tabla de Tarifas: Precios por temporada
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tarifas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        id_alojamiento INTEGER,
        temporada TEXT NOT NULL, -- 'alta', 'media', 'baja'
        precio_noche REAL NOT NULL,
        FOREIGN KEY (id_alojamiento) REFERENCES alojamientos(id)
    );
    """)

    # Tabla de Reservas: Se mantiene similar, pero ahora enlaza a alojamientos.id
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reservas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        id_alojamiento INTEGER NOT NULL,
        fecha_inicio TEXT NOT NULL,
        fecha_fin TEXT NOT NULL,
        nombre_cliente TEXT NOT NULL,
        FOREIGN KEY (id_alojamiento) REFERENCES alojamientos(id)
    );
    """)

    print("Tablas creadas con éxito.")

    # --- Poblar las tablas con la información del PDF ---

    # Datos extraídos del PDF 'Informacion Rosaleda.pdf'
    alojamientos_data = [
        ('Bungalow Luxe', 5, False, 'Cocina equipada, aire acondicionado, TV, terraza, salón, baño y jardín. Ropa de cama incluida (no toallas).'),
        ('Parcela Grande', 6, True, 'Parcela para hasta 6 personas. Coche incluido dentro. Se permite cocinar con gas o plancha.')
    ]
    cursor.executemany("INSERT INTO alojamientos (nombre, capacidad_max_personas, mascotas_permitidas, descripcion) VALUES (?, ?, ?, ?)", alojamientos_data)

    tarifas_data = [
        (1, 'alta', 165.00),   # Bungalow Luxe
        (1, 'media', 135.00),  # Bungalow Luxe (promedio de 125-145)
        (1, 'baja', 102.50),   # Bungalow Luxe (promedio de 95-110)
        (2, 'alta', 47.50),    # Parcela Grande (promedio de 45-50)
        (2, 'media', 35.00),   # Parcela Grande (promedio de 32-38)
        (2, 'baja', 27.50)     # Parcela Grande (promedio de 25-30)
    ]
    cursor.executemany("INSERT INTO tarifas (id_alojamiento, temporada, precio_noche) VALUES (?, ?, ?)", tarifas_data)
    
    # Añadimos algunas reservas de ejemplo consistentes
    reservas_data = [
        (1, '2025-07-10', '2025-07-20', 'Carlos Sánchez'), # Bungalow Luxe
        (2, '2025-08-01', '2025-08-15', 'Laura Jiménez')   # Parcela Grande
    ]
    cursor.executemany("INSERT INTO reservas (id_alojamiento, fecha_inicio, fecha_fin, nombre_cliente) VALUES (?, ?, ?, ?)", reservas_data)

    print("Datos insertados con éxito.")

    # Guardar cambios y cerrar conexión
    conn.commit()
    conn.close()
    print(f"Base de datos '{DB_FILE}' creada y poblada correctamente.")


if __name__ == "__main__":
    crear_base_de_datos()
